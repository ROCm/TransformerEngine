/*************************************************************************
 * Copyright (c) 2023-2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * License for AMD contributions = MIT. See LICENSE for more information
 ************************************************************************/
#include <type_traits>
#include <transformer_engine/gemm.h>
#include <transformer_engine/transformer_engine.h>
#include <map>
#include <unistd.h>
#include <vector>
#include <forward_list>
#include <mutex>
#include <unordered_map>
#include <sstream>
#include <fstream>
#include <chrono>
#include <optional>
#include <hipblaslt/hipblaslt.h>

#include <iostream>
#include <cstdlib>
#include <string>
#include <cstdint>

#include "../common.h"
#include "../util/vectorized_pointwise.h"
#include "../util/logging.h"

namespace transformer_engine {

namespace {

template<typename T> 
struct CacheEntry {
  T value;
  hipEvent_t event;

  constexpr CacheEntry() : value(), event(nullptr) {}
  
  bool isValid() const { return event != nullptr; }

  bool isAvailable() const
  {
    if (event == nullptr)
      return false;

    hipError_t err = hipEventQuery(event);
    if (err == hipSuccess)
    {
      return true;
    }
    else if (err == hipErrorNotReady)
    {
      return false;
    }
    else
    {
      NVTE_ERROR("Invalid event: err=", std::to_string(err), " ", hipGetErrorString(err));
      return false;
    }
  }
};

template<typename T, typename K> 
class ObjCache {
public:
  using Data = std::unordered_map<K, std::unordered_map<hipStream_t, CacheEntry<T>>>;
  static constexpr CacheEntry<T> invalidEntry{};

  const CacheEntry<T>& get(const K& key, const hipStream_t stream) const
  {
    auto key_itr = data.find(key); 
    if (key_itr == data.end())
      return invalidEntry;

    auto key_item = key_itr->second;

    if (auto itr = key_item.find(stream); itr != key_item.end())
      return itr->second;

    return invalidEntry;
  }

  CacheEntry<T> acquire(const K& key, hipStream_t stream, bool get_available = true)
  {
    auto key_itr = data.find(key); 
    if (key_itr == data.end())
      return invalidEntry;

    auto key_item = key_itr->second;

    if (auto itr = key_item.find(stream); itr != key_item.end())
    {
      auto ret = itr->second;
      key_item.erase(itr);
      return ret;
    }
    
    if (!get_available)
      return invalidEntry;

    for (auto itr = key_item.begin(); itr != key_item.end(); ++itr) {
      if (itr->second.isAvailable()) {
        auto ret = itr->second;
        key_item.erase(itr);
        return ret;
      }
    }
    return invalidEntry;
  }

  void set(const K& key, hipStream_t stream, const CacheEntry<T>& item)
  { 
    data[key][stream] = item; 
  }

  ObjCache(void (*a_offload)(const Data&)): offload(a_offload) {}

  ~ObjCache()
  {
    if (!data.empty() && offload != nullptr)
    {
      offload(data);
    }
  }

protected:
  void (*offload)(const Data&);
  Data data;
};

template<typename T, typename K>
class ObjPool: public ObjCache<T, K> {
  public:
    const CacheEntry<T>& get(const K& key, const hipStream_t stream) const
    {
      std::lock_guard<std::mutex> lock(mt);
      return ObjCache<T, K>::get(key, stream);
    }

    CacheEntry<T> acquire(const K& key, const hipStream_t stream, bool get_available = true)
    {
      std::lock_guard<std::mutex> lock(mt);
      return ObjCache<T, K>::acquire(key, stream, get_available);
    }

    void store(const typename ObjCache<T, K>::Data &cache)
    {
      std::lock_guard<std::mutex> lock(mt);
      for (const auto &it: cache)
      {
        for (const auto &it2: it.second)
        {
          ObjCache<T, K>::set(it.first, it2.first, it2.second);
        }
      }
    }

  ObjPool(): ObjCache<T, K>(nullptr) {}

  private:
    mutable std::mutex mt;
};
  

static hipDataType get_hipblaslt_dtype(const transformer_engine::DType t) {
  switch (t) {
    case DType::kFloat16:
      return HIP_R_16F;
    case DType::kFloat32:
      return HIP_R_32F;
    case DType::kBFloat16:
      return HIP_R_16BF;
#if HIP_VERSION >= 60300000
    case DType::kFloat8E4M3:
      return te_fp8_fnuz() ? HIP_R_8F_E4M3_FNUZ : HIP_R_8F_E4M3;
    case DType::kFloat8E5M2:
      return te_fp8_fnuz() ? HIP_R_8F_E5M2_FNUZ: HIP_R_8F_E5M2;
#else
    case DType::kFloat8E4M3:
      return HIP_R_8F_E4M3_FNUZ;
    case DType::kFloat8E5M2:
      return HIP_R_8F_E5M2_FNUZ;
#endif
    default:
      NVTE_ERROR("Invalid type");
  }
}

//TODO: unified with cublaslt_gemm.cu
struct GemmParam {
  void *A;
  void *B;
  cublasOperation_t transA;
  cublasOperation_t transB;
  transformer_engine::DType Atype;
  transformer_engine::DType Btype;
  void *A_scale_inv;
  void *B_scale_inv;
  int lda;
  int ldb;

  GemmParam(cublasOperation_t transA, cublasOperation_t transB)
      : A(nullptr),
        B(nullptr),
        transA(transA),
        transB(transB),
        Atype(transformer_engine::DType::kNumTypes),
        Btype(transformer_engine::DType::kNumTypes),
        A_scale_inv(nullptr),
        B_scale_inv(nullptr),
        lda(0),
        ldb(0) {}
};

GemmParam CanonicalizeGemmInput(const transformer_engine::Tensor &A, const cublasOperation_t transA,
                                const transformer_engine::Tensor &B, const cublasOperation_t transB,
                                const int k, const int lda, const int ldb) {
  using namespace transformer_engine;
  NVTE_CHECK(A.scaling_mode == B.scaling_mode,
             "Inputs A and B to GEMM need to have the same scaling mode!");
  NVTE_CHECK(A.has_data() || A.has_columnwise_data(), "Input A does not hold any data!");
  NVTE_CHECK(B.has_data() || B.has_columnwise_data(), "Input B does not hold any data!");
  GemmParam ret(transA, transB);

  // Transpose mode with column-major ordering
  bool is_A_transposed = transA == CUBLAS_OP_T;
  bool is_B_transposed = transB == CUBLAS_OP_T;

  ret.lda = lda;
  ret.ldb = ldb;

  if (is_tensor_scaling(A.scaling_mode)) {
    ret.A = A.data.dptr;
    ret.A_scale_inv = A.scale_inv.dptr;
    if (is_A_transposed) {
      ret.Atype = A.data.dtype;
    } else {
      ret.Atype = A.has_columnwise_data() ? A.columnwise_data.dtype : A.data.dtype;
      if (is_fp8_dtype(ret.Atype)) {
        // Hopper and Ada - we need to use columnwise_data and change transA
        NVTE_CHECK(A.has_columnwise_data(), "Input A is not suitable for columnwise usage!");
        ret.A = A.columnwise_data.dptr;
        ret.transA = CUBLAS_OP_T;
        ret.A_scale_inv = A.columnwise_scale_inv.dptr;
        ret.lda = k;
      }
    }
    ret.B = B.data.dptr;
    ret.B_scale_inv = B.scale_inv.dptr;
    if (is_B_transposed) {
      ret.Btype = B.has_columnwise_data() ? B.columnwise_data.dtype : B.data.dtype;
      if (is_fp8_dtype(ret.Btype)) {
        // Hopper and Ada - we need to use columnwise_data and change transA
        NVTE_CHECK(B.has_columnwise_data(), "Input B is not suitable for columnwise usage!");
        ret.B = B.columnwise_data.dptr;
        ret.transB = CUBLAS_OP_N;
        ret.B_scale_inv = B.columnwise_scale_inv.dptr;
        ret.ldb = k;
      }
    } else {
      ret.Btype = B.data.dtype;
    }
  } else {
    // If not tensor scaling (which includes also high precision types), we need to
    // use the proper version of data
    // We leave the transA/B values as is, since Blackwell supports transposes
    ret.A = is_A_transposed ? A.data.dptr : A.columnwise_data.dptr;
    ret.Atype = is_A_transposed ? A.data.dtype : A.columnwise_data.dtype;
    ret.A_scale_inv = is_A_transposed ? A.scale_inv.dptr : A.columnwise_scale_inv.dptr;
    ret.B = is_B_transposed ? B.columnwise_data.dptr : B.data.dptr;
    ret.Btype = is_B_transposed ? B.columnwise_data.dtype : B.data.dtype;
    ret.B_scale_inv = is_B_transposed ? B.columnwise_scale_inv.dptr : B.scale_inv.dptr;
  }
  return ret;
}


static class HandlePool {
public:
  hipblasLtHandle_t get(int device_id) 
  {
    std::lock_guard<std::mutex> lock(mt);

    if (pool.empty())
    {
      int device_count = 0; 
      NVTE_CHECK_CUDA(hipGetDeviceCount(&device_count));
      pool.resize(device_count);
      return nullptr;
    }

    if (!pool[device_id].empty())
    {
      hipblasLtHandle_t h = pool[device_id].front();
      pool[device_id].pop_front();
      return h;
    }

    return nullptr;
  }

  hipblasLtHandle_t obtain(int device_id) 
  {
    hipblasLtHandle_t h = get(device_id);
    if (h == nullptr)
    {
      NVTE_CHECK_HIPBLASLT(hipblasLtCreate(&h));
    }
    return h;
  }

  void store(const std::vector<hipblasLtHandle_t>& handles)
  {
    std::lock_guard<std::mutex> lock(mt);
    if (pool.empty())
    {
      std::cout << "[ERROR] Attempt to store handles to invalid pool" << std::endl;
    }
    for (unsigned int i=0; i<pool.size(); i++)
    {
      if (handles[i] != nullptr)
      {
        pool[i].push_front(handles[i]);
      }
    }
  }

  ~HandlePool() {
#if DESTROY_HIPBLASLT_HANDLES_POOL
    std::lock_guard<std::mutex> lock(mt);
    for (auto & hlist : pool)
    {
      for (auto & h : hlist)
      {
        hipblasLtDestroy(h);
      }
    }
    pool.clear();
#endif
  }

  inline size_t get_size() const
  {
    return pool.size();
  }

private:
  std::mutex mt;
  using Pool = std::vector<std::forward_list<hipblasLtHandle_t>>;
  // Order of destructors between thread_local and global is not actually guaranteed
  // As a simple w/a make pool storage "leaky"
  // Just do not destruct it and do not destroy hipbladLt handles
  // Let OS deal with it on application exit
#if DESTROY_HIPBLASLT_HANDLES_POOL
  Pool pool;
#else
  Pool &pool = *new Pool();
#endif
} handle_pool;


thread_local static class HandleCache {
public:
  hipblasLtHandle_t get(int device_id) const
  {
    return d.empty() ? nullptr : d[device_id];
  }

  hipblasLtHandle_t obtain(int device_id)
  {
    hipblasLtHandle_t h = get(device_id);
    if (h)
    {
      return h;
    }
    h = handle_pool.obtain(device_id);
    set(device_id, h);
    return h;
  }

  void set(int device_id, hipblasLtHandle_t h) 
  { 
    if (d.empty())
    {
      d.resize(handle_pool.get_size());
    }
    d[device_id] = h;
  }

  ~HandleCache()
  {
    if (!d.empty())
    {
      handle_pool.store(d);
    }
  }

private:
  std::vector<hipblasLtHandle_t> d;
} cached_handles;


class csv_helper
{
public:
  struct start {};
  struct end {};

  csv_helper(std::ostream& os, char sep_val) : m_os{ os }, m_sep_val(sep_val), m_start(true), m_sep("") {}

  csv_helper& operator << (const start&)
  {
    m_start = true;
    return *this;
  }

  csv_helper& operator << (const end&)
  {
    m_sep="";
    m_start = false;
    return *this;
  }

  template< typename T>
  csv_helper& operator<<(const T& v)
  {
    m_os << m_sep << v;
    if (m_start)
    {
      m_start = false;
      m_sep = m_sep_val;
    }
    return *this;
  }

private:
  std::ostream& m_os;
  char m_sep_val;
  bool m_start;
  std::string m_sep;
};


template<typename T>
class NameMapper
{
public:
  NameMapper(const std::unordered_map<T, std::string_view>& name_map): map(name_map) {}
  const std::string_view &getName(const T &val) {
    return map.at(val);
  }
  T getValue(const std::string& name, const char *label="", std::function<bool(const T&)> filter = nullptr)
  {
    for (auto iter = map.begin(); iter != map.end(); ++iter)
    {
      if ((name == iter->second) && (!filter || filter(iter->first))) return iter->first;
    }
    NVTE_ERROR("Invalid ", label, " name: ", name);
  }
protected: 
  const std::unordered_map<T, std::string_view> &map;
};

static std::unordered_map<hipDataType, std::string_view> type_name_map = {
  {HIP_R_32F, "float32"},
  {HIP_R_16F, "float16"},
  {HIP_R_16BF, "bfloat16"},
  {HIP_R_8F_E4M3_FNUZ, "float8e4m3"},
  {HIP_R_8F_E5M2_FNUZ, "float8e5m2"},
#if HIP_VERSION >= 60300000
  {HIP_R_8F_E4M3, "float8e4m3"},
  {HIP_R_8F_E5M2, "float8e5m2"},
#endif
};
static NameMapper<hipDataType> typeNameMapper(type_name_map);

static std::unordered_map<hipblasOperation_t, std::string_view> trans_name_map = {
  {HIPBLAS_OP_N, "N"},
  {HIPBLAS_OP_T, "T"}
};
static NameMapper<hipblasOperation_t> transposeNameMapper(trans_name_map);

static std::unordered_map<hipblasLtEpilogue_t, std::string_view> epi_name_map = {
  {HIPBLASLT_EPILOGUE_DEFAULT, "-"},
  {HIPBLASLT_EPILOGUE_BIAS, "bias"},
  {HIPBLASLT_EPILOGUE_GELU_AUX, "geluaux"},
  {HIPBLASLT_EPILOGUE_GELU_AUX_BIAS, "geluauxbias"},
  {HIPBLASLT_EPILOGUE_DGELU, "dgelu"},
  {HIPBLASLT_EPILOGUE_DGELU_BGRAD, "dgelubgrad"},
  {HIPBLASLT_EPILOGUE_BGRADB, "bgradb"}
};
static NameMapper<hipblasLtEpilogue_t> epilogueNameMapper(epi_name_map);

static std::unordered_map<hipblasComputeType_t, std::string_view> comp_name_map = {
  {HIPBLAS_COMPUTE_32F, "f32"}
};
static NameMapper<hipblasComputeType_t> computeNameMapper(comp_name_map);

static class GemmAlgoCache {
public:
  struct Key {
    int deviceCap;
    hipDataType a_type, b_type, d_type, bias_type;
    int m, n, k;
    int lda, ldb, ldd;
    hipblasOperation_t transa, transb;
    //fp8_scale is int instead of hipblasLtMatmulMatrixScale_t for compatibility with old hipblasLt
    int fp8_scale;
    hipblasLtEpilogue_t epilogue;

    Key(int deviceCap_,
        hipDataType a_type_, hipDataType b_type_,
        hipDataType d_type_, hipDataType bias_type_,
        int m_, int n_, int k_, int lda_, int ldb_, int ldd_,
        hipblasOperation_t transa_, hipblasOperation_t transb_,
        int fp8_scale_, hipblasLtEpilogue_t epilogue_):
        deviceCap(deviceCap_),
        a_type(a_type_), b_type(b_type_),
        d_type(d_type_), bias_type(bias_type_),
        m(m_), n(n_), k(k_), lda(lda_), ldb(ldb_), ldd(ldd_),
        transa(transa_), transb(transb_),
        fp8_scale(fp8_scale_), epilogue(epilogue_) {}

    Key() {}

    bool operator==(const Key &val) const
    {
      return ((deviceCap == val.deviceCap)
      && (a_type == val.a_type) && (b_type == val.b_type)
      && (d_type == val.d_type) && (bias_type == val.bias_type)
      && (m == val.m) && (n == val.n) && (k == val.k)
      && (lda == val.lda) && (ldb == val.ldb) && (ldd == val.ldd)
      && (transa == val.transa) && (transb == val.transb)
      && (fp8_scale == val.fp8_scale) && (epilogue == val.epilogue) );
    }

    struct Comp
    {
      bool operator()(const Key& lhs, const Key& rhs) const
      {
        return ::std::string_view((const char*)&lhs, sizeof(lhs)) < ::std::string_view((const char*)&rhs, sizeof(rhs));
      }
    };
  };

  void init()
  {
    std::lock_guard<std::mutex> lock(mt);
    int device_count = 0; 
    NVTE_CHECK_CUDA(hipGetDeviceCount(&device_count));
    dev_cap.resize(device_count);
    for (int i=0; i<device_count; i++)
    {
      hipDeviceProp_t prop;
      NVTE_CHECK_CUDA(hipGetDeviceProperties(&prop, i));
      dev_cap[i] = prop.major*100 + prop.minor;
    }
    load_();
    save_();
  }

  inline int device_cap(int device_id)
  {
    if (dev_cap.empty())
      init();
    return dev_cap[device_id];
  }

  struct Algo {
    std::optional<hipblasLtMatmulAlgo_t> algo;
    int64_t algoId;
    int index;
    size_t ws_size_min;
    size_t ws_size_max;
    Algo(): algo(), index(-1), algoId(), ws_size_min(0), ws_size_max(0) {}
    Algo(int idx, int64_t id, size_t ws_min, size_t ws_max): algo(), index(idx), algoId(id), ws_size_min(ws_min), ws_size_max(ws_max) {}
    inline bool hasId() { return index>=0; } const
    static inline int64_t getAlgoId(const hipblasLtMatmulAlgo_t &algo)
    {
      return *(const int64_t*)&algo;
    }
  };

  bool find(const Key &cfg, size_t ws_size, Algo &algo)
  {
    std::lock_guard<std::mutex> lock(mt);
    if (auto *pentry = find_(cfg, ws_size, ws_size); pentry != nullptr)
    {
      algo = *pentry;
      return true;
    }
    return false;
  }

  void store(const Key &cfg, const Algo &algo)
  {
    size_t ws_size_min = algo.ws_size_min;
    size_t ws_size_max = algo.ws_size_max;
    NVTE_CHECK(ws_size_max >= ws_size_min, "Invalid WS size");
    std::lock_guard<std::mutex> lock(mt);

    //Remove overlapping with existing entries;
    while (auto* pentry = find_(cfg, ws_size_min, ws_size_max)) {
      if (pentry->ws_size_min <= ws_size_min && pentry->ws_size_max >= ws_size_max)
      {
        *pentry = algo;
        save_();
        return;
      }

      if (ws_size_max > pentry->ws_size_max)
      {
        ws_size_min = pentry->ws_size_max + 1;
      }
      else if (ws_size_min < pentry->ws_size_min)
      {
        ws_size_max = pentry->ws_size_min - 1;
      }
      else
      {
        //Should never be here
        NVTE_ERROR("Cannot merge WS size range");
      }
    }

    //Merge to adjusted entry if possible
    auto* pentry = find_(cfg, ws_size_min - 1, ws_size_min);
    if (pentry && pentry->algoId == algo.algoId)
    {
      pentry->algo = algo.algo;
      pentry->ws_size_max = ws_size_max;
      save_();
    }
    else
    {
      auto it = d.emplace(cfg, algo);
      it->second.ws_size_min = ws_size_min;
      it->second.ws_size_max = ws_size_max;
      save_(it->first, it->second);
    }
  }

protected:

  Algo* find_(const Key &cfg, size_t ws_min, size_t ws_max)
  {
    const auto key_range = d.equal_range(cfg);
    for (auto i = key_range.first; i != key_range.second; i++)
    {
      if (ws_min <= i->second.ws_size_max && ws_max >= i->second.ws_size_min)
      {
        return &i->second;
      }
    }
    return nullptr;
  }

  void header_(std::ostream& ofs)
  {
    csv_helper fs(ofs, csv_sep);
    fs << "dev_cap" << "m" << "n"  << "k" << "trans_a" << "trans_b" 
    << "type_a" << "type_b" << "type_d" << "bias_type" 
    << "lda" << "ldb" << "ldd" << "fp8_scale" << "epi" << "comp" << "scale"
    << "ws_min" << "ws_max" << "algo_id" << "aidx";
  }
  
  void load_()
  {
    const char* env = std::getenv("TE_HIPBLASLT_ALGO_LOAD");
    if (env == nullptr || env[0] == '\0')
    {
      return;
    }
    std::ifstream ifs{env};
    if (!ifs.is_open())
    {
      std::cerr << "Could not load autotune results storage " << env << "\n";
      return;
    }
    std::cout << "Loading autotune results from " << env << "\n";

    Key cfg;
    std::string line;
    std::getline(ifs, line); // the first line with legend
    {
      std::ostringstream hline;
      header_(hline);
      if (hline.str() != line) {
        std::cerr << "Incorrect algo storage legend. Expected " << hline.str() << "\n";
        return;
      }
    }

    while(std::getline(ifs, line)) 
    {
      line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));
      if (auto pos = line.find_last_not_of(" \t\n\r\f\v"); pos != std::string::npos)
      {
        line.resize(pos+1);
      }
      if (line.empty() || line[0] == '#') continue;
      std::istringstream is(line);
      char c;
      std::string type_a, type_b, type_d, bias_type, trans_a, trans_b, epi, comp, scale;
      int64_t algo_id;
      int algo_idx;
      size_t ws_min, ws_max;

      is >> std::skipws;
      is >> cfg.deviceCap >> c >> cfg.m >> c >> cfg.n >> c >> cfg.k >> c;

      //Filter out entries for devices not presented on the curent system
      bool b_found = false;
      for (int i=0; i<dev_cap.size(); i++)
      {
        if (dev_cap[i] == cfg.deviceCap)
        {
          b_found = true;
          break;
        }
      }
      if (!b_found) continue;

      std::getline(is, trans_a, csv_sep);
      std::getline(is, trans_b, csv_sep);
      std::getline(is, type_a, csv_sep);
      std::getline(is, type_b, csv_sep);
      std::getline(is, type_d, csv_sep);
      std::getline(is, bias_type, csv_sep);
      is >> cfg.lda >> c >> cfg.ldb >> c >> cfg.ldd >> c >> cfg.fp8_scale >> c;
      std::getline(is, epi, csv_sep);
      std::getline(is, comp, csv_sep);
      std::getline(is, scale, csv_sep);
      is >> ws_min >> c >> ws_max >> c >> algo_id >> c >> algo_idx;
  
      if (is.bad())
      {
        std::cerr << "Parsing CSV line failed: " << line << "\n";
        return;
      }

      if (ws_min > ws_max)
      {
        std::cout << "[WARNING] Invalid WS size at " << line << "\n";
        continue;
      }

      //Check and filter out compute and scale types
      if (computeNameMapper.getValue(comp, "comp") != HIPBLAS_COMPUTE_32F ||
        typeNameMapper.getValue(scale, "scale") != HIP_R_32F)
      {
        continue;
      }

#if HIPBLASLT_VERSION_MAJOR > 0 || HIPBLASLT_VERSION_MINOR >= 15
      if (cfg.fp8_scale < 0 || cfg.fp8_scale >= (int)HIPBLASLT_MATMUL_MATRIX_SCALE_END)
#else
      if (cfg.fp8_scale != 0)
#endif
      {
        continue;
      }

#if HIP_VERSION >= 60300000
      auto fp8_filter = te_fp8_fnuz()
                            ? [](const hipDataType& val) 
                                { return (val != HIP_R_8F_E4M3 && val != HIP_R_8F_E5M2); }
                            : [](const hipDataType& val) {
                                return (val != HIP_R_8F_E4M3_FNUZ && val != HIP_R_8F_E5M2_FNUZ);
                              };
#else
      auto fp8_filter = nullptr;
#endif

      cfg.a_type = typeNameMapper.getValue(type_a, "type_a", fp8_filter);
      cfg.b_type = typeNameMapper.getValue(type_b, "type_b", fp8_filter);
      cfg.d_type = typeNameMapper.getValue(type_d, "type_d", fp8_filter);
      cfg.bias_type = (bias_type == "-")
                          ? (hipDataType)-1
                          : typeNameMapper.getValue(bias_type, "bias_type", fp8_filter);

      cfg.transa = transposeNameMapper.getValue(trans_a, "trans_a");
      cfg.transb = transposeNameMapper.getValue(trans_b, "trans_b");

      cfg.epilogue = epilogueNameMapper.getValue(epi, "epi");

      if (find_(cfg, ws_min, ws_max))
      {
          std::cout << "[WARNING] Duplicated/overlapped entry in algo cache\n";
          continue;
      }

      d.emplace(cfg, Algo(algo_idx, algo_id, ws_min, ws_max));
    }
  }

  bool can_save_(bool reopen = false)
  {
    if (!save_fs)
    {
      const char* temp = std::getenv("TE_HIPBLASLT_ALGO_SAVE");
      if (temp == nullptr || temp[0] == '\0')
      {
        return false;
      }

      save_fs_name = temp;

      pid_t pid = getpid();

      size_t pos = 0;
      while ((pos = save_fs_name.find("%i", pos)) != std::string::npos) {
        save_fs_name.replace(pos, 2, std::to_string(pid));
      }

      save_fs = std::make_unique<std::ofstream>();
      std::cout << "Saving autotune results to " << save_fs_name << "\n";
    }

    if (reopen)
    {
      if (save_fs->is_open())
      {
        save_fs->close();
      }
      save_fs->open(save_fs_name, std::ios_base::trunc);
    }

    if (save_fs->is_open() && !save_fs->bad())
    {
      return true;
    }
    else
    {
      if (reopen) std::cerr << "Could not open autotune results storage " << save_fs_name << "\n";
      return false;
    }
  }

  void save_()
  {
    if (!can_save_(true))
    {
      return;
    }
    header_(*save_fs);
    *save_fs << "\n";

    for (const auto &elem: d)
    {
      save_(elem.first, elem.second);
    }
  }

  void save_(const Key &cfg, const Algo &algo)
  {
    if (!can_save_())
    {
      return;
    }
    csv_helper csv(*save_fs, csv_sep);
    csv << cfg.deviceCap << cfg.m << cfg.n << cfg.k 
      << transposeNameMapper.getName(cfg.transa) << transposeNameMapper.getName(cfg.transb)
      << typeNameMapper.getName(cfg.a_type) << typeNameMapper.getName(cfg.b_type) << typeNameMapper.getName(cfg.d_type)
      << ((cfg.bias_type == (hipDataType)-1) ? "-" : typeNameMapper.getName(cfg.bias_type))
      << cfg.lda << cfg.ldb << cfg.ldd << cfg.fp8_scale << epilogueNameMapper.getName(cfg.epilogue)
      << computeNameMapper.getName(HIPBLAS_COMPUTE_32F) << typeNameMapper.getName(HIP_R_32F)
      << algo.ws_size_min << algo.ws_size_max << algo.algoId << algo.index << csv_helper::end() << "\n";
  }

private:
  std::vector<int> dev_cap;
  constexpr static char csv_sep = ','; 
  std::unique_ptr<std::ofstream> save_fs;
  std::string save_fs_name;
  std::mutex mt;
  /* Map of problem config to tuple of ws_size and Algo
   * When searching, elements matching Key are filtered 
   * for requested WS size be between Algo.ws_size and pair.first
   */
  std::multimap<Key, Algo, Key::Comp> d;
} algoCache;

static inline int getIntEnv(const char *name, int defval, int minval)
{
  int val = defval;
  const char* env = std::getenv(name);
  if (env != nullptr && env[0] != '\0')
  {
     val = atoi(env);
     if (val < minval)
     {
        val = minval;
     }
  }
  return val;
}


/* Warning: only call once per device!
 * When calling nvte_multi_stream_cublas_gemm with hipblaslt backend
 * need to create multiple handles corresponding to compute_streams
 * to avoid a handle be used by multi-streams concurrently.
 */
static void init_hipblaslt_handles(hipblasLtHandle_t* hipblaslt_handles) {
  NVTE_CHECK(hipblaslt_handles != nullptr);
  for (int i = 0; i < num_streams; i++) {
    NVTE_CHECK_HIPBLASLT(hipblasLtCreate(&hipblaslt_handles[i]));
  }
}


void hipblaslt_gemm(const Tensor *inputA,
                    const Tensor *inputB,
                    Tensor *outputD,
                    const Tensor *inputBias,
                    Tensor *outputPreGelu,
                    int m, int n, int k,
                    int lda, int ldb, int ldd,
                    hipblasOperation_t transa,
                    hipblasOperation_t transb,
                    bool grad,
                    void* workspace,
                    size_t workspaceSize,
                    bool accumulate,
                    bool use_split_accumulator,
                    int math_sm_count,
                    int m_split,
                    int n_split,
                    bool gemm_producer,
                    const Tensor *inputCounter,
                    hipStream_t stream,
                    hipblasLtHandle_t handle
) {
  // Return immediately if GEMM is trivial
  if (m <= 0 || n <= 0) {
    return;
  }
  NVTE_CHECK(k > 0);

  const GemmParam &param = CanonicalizeGemmInput(*inputA, transa, *inputB, transb, k, lda, ldb);

  bool nvte_log_gemm_config = false;
  if (const char* env_p = std::getenv("NVTE_LOG_GEMM_CONFIG") ) {
      nvte_log_gemm_config = (strcmp(env_p, "1") == 0);
  }

  if (nvte_log_gemm_config) {
    const bool use_fp8 = is_fp8_dtype(param.Atype) || is_fp8_dtype(param.Btype);
    const bool a_tensor = is_tensor_scaling(inputA->scaling_mode);
    const bool a_block  = is_block_scaling(inputA->scaling_mode);

    std::cout << "m=" << m << " k=" << k << " n=" << n 
        << " transa=" << (param.transA == HIPBLAS_OP_T ? "T" : "N")
        << " transb=" << (param.transB == HIPBLAS_OP_T ? "T" : "N")
        << " A_type=" << (int)(param.Atype)
        << " B_type=" << (int)(param.Btype)
        << " D_type=" << (int)outputD->data.dtype
        << " bias_type=" << (int)inputBias->data.dtype
        << " grad=" << grad
        << " bias=" << (inputBias->data.dptr != nullptr)
        << " gelu=" << (outputPreGelu->data.dptr != nullptr)
        << " use_fp8=" << use_fp8
        << " scale_mode=" << (a_tensor ? "tensor" : a_block ? "mxfp8" : "unsupported")
        << " accumulate=" << accumulate
        << std::endl;
  }
  
  void *D = outputD->data.dptr;
  void *C = D;
  void *D_scale = outputD->scale.dptr;
  void *D_amax = outputD->amax.dptr;
  void *bias_ptr = inputBias->data.dptr;
  const bool bias = bias_ptr != nullptr;
  void *pre_gelu_out = outputPreGelu->data.dptr;
  const bool gelu = pre_gelu_out != nullptr;
  const bool use_fp8 = is_fp8_dtype(param.Atype) || is_fp8_dtype(param.Btype);

  const hipDataType A_type = get_hipblaslt_dtype(param.Atype);
  const hipDataType B_type = get_hipblaslt_dtype(param.Btype);
  const hipDataType D_type = get_hipblaslt_dtype(outputD->data.dtype);
  const hipDataType bias_type = get_hipblaslt_dtype(inputBias->data.dtype);
  // const hipblasltDatatype_t aux_type = get_hipblaslt_dtype(outputPreGelu->data.dtype);

  NVTE_CHECK(!is_fp8_dtype(param.Atype) || param.A_scale_inv != nullptr,
             "FP8 input to GEMM requires inverse of scale!");
  NVTE_CHECK(!is_fp8_dtype(param.Btype) || param.B_scale_inv != nullptr,
             "FP8 input to GEMM requires inverse of scale!");

  // check consistency of arguments:
  // if fp8 is desired, context cannot be null
  // fp8 + gelu fusion + fp8 aux is unavailable right now.
  if (use_fp8) {
    NVTE_CHECK(!gelu, "fp8 gemm + gelu fusion is unavailable right now!");
  }
  if (is_fp8_dtype(outputD->data.dtype)) {
    NVTE_CHECK(!accumulate, "Accumulation mode not supported with FP8 GEMM output!");
  }

  float one = 1.0;
  float zero = 0.0;
  float beta = (accumulate) ? one : zero;

  int device_id;
  NVTE_CHECK_CUDA(hipGetDevice(&device_id));

  if (handle == nullptr) {
    handle = cached_handles.get(device_id);
    if (handle == nullptr)
    {
      handle = cached_handles.obtain(device_id);
    }
  }

  hipblasLtMatmulDesc_t       operationDesc = nullptr;
  hipblasLtMatrixLayout_t     Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr, Ddesc = nullptr;
  hipblasLtMatmulPreference_t preference = nullptr;
  hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;

  int64_t ld_gelumat = (int64_t) ldd;

  // default to tf32 except for e5m2 inputs where the config is not supported
  hipblasComputeType_t gemm_compute_type = HIPBLAS_COMPUTE_32F;

  // Create matrix descriptors. Not setting any extra attributes.
  NVTE_CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&Adesc, A_type,
                                                   param.transA == HIPBLAS_OP_N ? m : k,
                                                   param.transA == HIPBLAS_OP_N ? k : m,
                                                   param.lda));
  NVTE_CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&Bdesc, B_type,
                                                   param.transB == HIPBLAS_OP_N ? k : n,
                                                   param.transB == HIPBLAS_OP_N ? n : k,
                                                   param.ldb));
  NVTE_CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&Ddesc, D_type, m, n, ldd));
  Cdesc = Ddesc;

  NVTE_CHECK_HIPBLASLT(hipblasLtMatmulDescCreate(&operationDesc, gemm_compute_type, HIP_R_32F));
  NVTE_CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                                       &param.transA, sizeof(param.transA)));
  NVTE_CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_TRANSB,
                                                       &param.transB, sizeof(param.transB)));

  // set fp8 attributes -- input and output types should already be set to fp8 as appropriate
  // Note: gelu fusion isn't available right now, and we don't need
  // amax(D) either (next op is high precision).
#if HIPBLASLT_VERSION_MAJOR > 0 || HIPBLASLT_VERSION_MINOR >= 15
    hipblasLtMatmulMatrixScale_t scaling_mode;
#else
    constexpr int scaling_mode = 0;
#endif
  if (use_fp8) {
    // Split accumulator.
    const int8_t fastAccuMode = (use_split_accumulator) ? 0 : 1;
    /*
    NVTE_CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                     HIPBLASLT_MATMUL_DESC_FAST_ACCUM, //TODO: We don't have fast accum mode yet
                                                     &fastAccuMode,
                                                     sizeof(fastAccuMode)));
    */
    if ((is_delayed_tensor_scaling(inputA->scaling_mode) &&
         is_delayed_tensor_scaling(inputB->scaling_mode))) {
#if HIPBLASLT_VERSION_MAJOR > 0 || HIPBLASLT_VERSION_MINOR >= 15
      scaling_mode = HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    } else if ((is_block_scaling(inputA->scaling_mode) && is_block_scaling(inputB->scaling_mode))) {
      scaling_mode = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
      NVTE_CHECK(!is_fp8_dtype(outputD->data.dtype), "FP8 output is not supported with block scaling mode.");
#endif
    } else {
      NVTE_ERROR("Not implemented scaling modes: " + to_string(inputA->scaling_mode) + " and  " +
                 to_string(inputB->scaling_mode) + ".");
    }
    NVTE_CHECK_HIPBLASLT(
        hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                        &param.A_scale_inv, sizeof(param.A_scale_inv)));
    NVTE_CHECK_HIPBLASLT(
        hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                        &param.B_scale_inv, sizeof(param.B_scale_inv)));
#if HIPBLASLT_VERSION_MAJOR > 0 || HIPBLASLT_VERSION_MINOR >= 15
    NVTE_CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(
        operationDesc, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaling_mode, sizeof(scaling_mode)));
    NVTE_CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(
        operationDesc, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaling_mode, sizeof(scaling_mode)));
#endif

    if (is_fp8_dtype(outputD->data.dtype)) {
      NVTE_CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(
        operationDesc, HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER, &D_scale, sizeof(D_scale)));
      NVTE_CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(
        operationDesc, HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER, &D_amax, sizeof(D_amax)));
    }
    if (bias) {
      NVTE_CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                       HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                                       &bias_type, sizeof(bias_type)));
    }
  }
  
  if (bias && gelu) {
    if (grad) {
      epilogue = HIPBLASLT_EPILOGUE_DGELU_BGRAD;
    } else {
      epilogue = HIPBLASLT_EPILOGUE_GELU_AUX_BIAS;
    }
    NVTE_CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                      HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                      &bias_ptr, sizeof(bias_ptr)));
    NVTE_CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(
                            operationDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
                            &pre_gelu_out, sizeof(pre_gelu_out)));
    NVTE_CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                      HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,
                                                      &ld_gelumat, sizeof(ld_gelumat)));
    // TODO: future enablement
    //const hipDataType aux_type = get_hipblaslt_dtype(outputPreGelu->data.dtype);
    //NVTE_CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(
    //  operationDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE, &aux_type, sizeof(aux_type)));
  } else if (bias) {
    if (grad) {
      // grad output is always input B
      epilogue = HIPBLASLT_EPILOGUE_BGRADB;
    } else {
      epilogue = HIPBLASLT_EPILOGUE_BIAS;
    }
    NVTE_CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                      HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                      &bias_ptr, sizeof(bias_ptr)));
  } else if (gelu) {
    if (grad) {
      epilogue = HIPBLASLT_EPILOGUE_DGELU;
    } else {
      epilogue = HIPBLASLT_EPILOGUE_GELU_AUX;
    }
    NVTE_CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(
                            operationDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
                            &pre_gelu_out, sizeof(pre_gelu_out)));
    NVTE_CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                     HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,
                                                     &ld_gelumat, sizeof(ld_gelumat)));
  }

  NVTE_CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                   HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                                   &epilogue, sizeof(epilogue)));

  GemmAlgoCache::Key gemm_cfg(algoCache.device_cap(device_id), A_type, B_type, D_type, 
    use_fp8 ? bias_type : (hipDataType)-1,
    m, n, k, param.lda, param.ldb, ldd, param.transA, param.transB, scaling_mode, epilogue );
  GemmAlgoCache::Algo cached_algo;
  if (algoCache.find(gemm_cfg, workspaceSize, cached_algo) == 0 || !cached_algo.algo.has_value())
  {
    int firstAlgo = getIntEnv("TE_HIPBLASLT_ALGO_SELECTION", 0, 0);
    int tuneLoopCount = getIntEnv("TE_HIPBLASLT_TUNING_RUN_COUNT", 0, 0);
    int algoTuneCount = 1;
    std::vector<hipblasLtMatmulHeuristicResult_t> algoArr;
    bool logTuning = getIntEnv("TE_HIPBLASLT_LOG_TUNING", 0, 0) != 0;

    if (tuneLoopCount)
    {
      /* HIPBLASLT may return hundreds of algos for some configs
       * Limit amount by default. User may override with env
       */
      static const int defaultAlgoCount = 16;
      algoTuneCount = getIntEnv("TE_HIPBLASLT_TUNING_ALGO_COUNT", defaultAlgoCount, 1);
    }
    algoTuneCount += firstAlgo;
    int algoTotalCount = cached_algo.hasId() ? std::max(algoTuneCount, (cached_algo.index + 1)) : algoTuneCount;
    algoArr.resize(algoTotalCount);

    NVTE_CHECK_HIPBLASLT(hipblasLtMatmulPreferenceCreate(&preference));
    NVTE_CHECK_HIPBLASLT(hipblasLtMatmulPreferenceSetAttribute(
                            preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                            &workspaceSize, sizeof(workspaceSize)));

    NVTE_CHECK_HIPBLASLT(hipblasLtMatmulAlgoGetHeuristic(handle, operationDesc, Adesc, Bdesc, Cdesc,
                                                    Ddesc, preference, algoTotalCount, algoArr.data(),
                                                    &algoTotalCount));
    algoArr.resize(algoTotalCount);

    NVTE_CHECK_HIPBLASLT(hipblasLtMatmulPreferenceDestroy(preference));

    //If cached algo exists in persistent storage we just need to find matching hipblasLtMatmulAlgo_t
    if (cached_algo.hasId())
    {
      int idx = (cached_algo.index < algoTotalCount) ? cached_algo.index : 0;
      for (int i=0; i<algoTotalCount; i++)
      {
        const auto &algo = algoArr[idx];
        if (algo.state == HIPBLAS_STATUS_SUCCESS)
        {
          if (cached_algo.algoId == cached_algo.getAlgoId(algo.algo))
          {
            cached_algo.algo = algo.algo;
            if (algo.workspaceSize != cached_algo.ws_size_min || idx != cached_algo.index)
            {
              cached_algo.ws_size_min = algo.workspaceSize;
              cached_algo.index = idx;
              algoCache.store(gemm_cfg, cached_algo);
            }
            break;
          }
        }
        idx = (idx + 1) % algoTotalCount;
      }
      if (logTuning && !cached_algo.algo.has_value())
      {
        std::cout << "[WARNING] Cannot find cached algoId " << cached_algo.algoId << " in hipBLASLt results" << std::endl;
      }
    }

    //No suitable entry in autotune cache or could not find matched algo in hipBLASLt results
    if (!cached_algo.algo.has_value())
    {

      int bestAlgo = -1;
      algoTuneCount = std::min(algoTuneCount, algoTotalCount);
      if (tuneLoopCount > 0)
      {
        if (logTuning)
          std::cout << "[INFO] Perform hipBLASLt algo selection on GPU" << device_id
                    << " in range [" << firstAlgo << "-" << (algoTuneCount - 1) << "] with "
                    << tuneLoopCount << " loops " << std::endl;

        NVTE_CHECK_CUDA(hipStreamSynchronize(stream));
        hipStream_t &profilingStream = stream; // Reuse the stream for profiling
        using tuning_clock = std::chrono::steady_clock;
        tuning_clock::now(); //the first call takes little longer so do it outside the loop
        tuning_clock::duration bestTime = tuning_clock::duration::max();

        for (int algo=firstAlgo; algo<algoTuneCount; algo++)
        {
            if (algoArr[algo].state != HIPBLAS_STATUS_SUCCESS)
            {
              continue;
            }
            // Warm-up call
            NVTE_CHECK_HIPBLASLT(hipblasLtMatmul(handle,
                                            operationDesc,
                                            static_cast<const void*>(&one),         /* alpha */
                                            param.A,                                      /* A */
                                            Adesc,
                                            param.B,                                      /* B */
                                            Bdesc,
                                            static_cast<const void*>(&beta),        /* beta */
                                            C,                                      /* C */
                                            Cdesc,
                                            D,                                      /* D */
                                            Ddesc,
                                            &algoArr[algo].algo,                    /* algo */
                                            workspace,                              /* workspace */
                                            workspaceSize,
                                            profilingStream));                       /* stream */
          NVTE_CHECK_CUDA(hipStreamSynchronize(profilingStream));

          //Profiling loop
          tuning_clock::time_point startTime = tuning_clock::now();
          for (int loop=0; loop<tuneLoopCount; loop++)
          {
            NVTE_CHECK_HIPBLASLT(hipblasLtMatmul(handle,
                                            operationDesc,
                                            static_cast<const void*>(&one),         /* alpha */
                                            param.A,                                      /* A */
                                            Adesc,
                                            param.B,                                      /* B */
                                            Bdesc,
                                            static_cast<const void*>(&beta),        /* beta */
                                            C,                                      /* C */
                                            Cdesc,
                                            D,                                      /* D */
                                            Ddesc,
                                            &algoArr[algo].algo,                    /* algo */
                                            workspace,                              /* workspace */
                                            workspaceSize,
                                            profilingStream));                       /* stream */
          }
          NVTE_CHECK_CUDA(hipStreamSynchronize(profilingStream));
          tuning_clock::duration algoTime = tuning_clock::now() - startTime; 
          if (algoTime < bestTime)
          {
            bestAlgo = algo;
            bestTime = algoTime;
          }
        }

        if (bestAlgo >= 0)
        {
          if (logTuning)
            std::cout << "[INFO] Select hipBLASLt algo " << bestAlgo << " with time "
                      << std::chrono::duration_cast<std::chrono::nanoseconds>(bestTime).count() / tuneLoopCount
                      << " ns" << std::endl;
        }
      }
      else if (firstAlgo < algoTuneCount)
      {
        bestAlgo = firstAlgo;
      }

      if (bestAlgo < 0) {
        NVTE_CHECK_HIPBLASLT(hipblasLtMatrixLayoutDestroy(Ddesc));
        NVTE_CHECK_HIPBLASLT(hipblasLtMatrixLayoutDestroy(Bdesc));
        NVTE_CHECK_HIPBLASLT(hipblasLtMatrixLayoutDestroy(Adesc));
        NVTE_CHECK_HIPBLASLT(hipblasLtMatmulDescDestroy(operationDesc));
        throw std::runtime_error("Unable to find any suitable algorithms");
      }
      cached_algo.algo = algoArr[bestAlgo].algo;
      cached_algo.index = bestAlgo;
      cached_algo.algoId = cached_algo.getAlgoId(algoArr[bestAlgo].algo);
      cached_algo.ws_size_min = algoArr[bestAlgo].workspaceSize;
      cached_algo.ws_size_max = workspaceSize;

      if (logTuning)
        std::cout << "[INFO] Use hipBLASLt algo [" << bestAlgo << "] " << cached_algo.algoId << std::endl;

      algoCache.store(gemm_cfg, cached_algo);
    }
  }

  // D = alpha * (A * B) + beta * C
  NVTE_CHECK_HIPBLASLT(hipblasLtMatmul(handle,
                                   operationDesc,
                                   static_cast<const void*>(&one),         /* alpha */
                                   param.A,                                      /* A */
                                   Adesc,
                                   param.B,                                      /* B */
                                   Bdesc,
                                   static_cast<const void*>(&beta),        /* beta */
                                   C,                                      /* C */
                                   Cdesc,
                                   D,                                      /* D */
                                   Ddesc,
                                   &cached_algo.algo.value(),              /* algo */
                                   workspace,                              /* workspace */
                                   workspaceSize,
                                   stream));                               /* stream */

  // Update FP8 scale-inv in output tensor
  // Note: This is a WAR for the case when we have fp8 output but D->scale_inv is not allocated.
  // TODO: Changing gemm interface so that D->scale_inv is allocated and the scale_inv can be
  // calculated here.
  if (is_fp8_dtype(outputD->data.dtype) && outputD->scale_inv.dptr) {
    update_tensor_scale_inv(outputD, stream);
  }

  NVTE_CHECK_HIPBLASLT(hipblasLtMatrixLayoutDestroy(Ddesc));
  NVTE_CHECK_HIPBLASLT(hipblasLtMatrixLayoutDestroy(Bdesc));
  NVTE_CHECK_HIPBLASLT(hipblasLtMatrixLayoutDestroy(Adesc));
  NVTE_CHECK_HIPBLASLT(hipblasLtMatmulDescDestroy(operationDesc));
}


typedef unsigned long long ServiceStreamKey;

ServiceStreamKey make_service_stream_key(const int device_id, const int cu_count) {
  return (static_cast<ServiceStreamKey>(device_id) << 32) | static_cast<ServiceStreamKey>(cu_count);
}

std::pair<int, int> parse_service_stream_key(const ServiceStreamKey &key) {
  int device_id = static_cast<int>(key >> 32);
  int cu_count = static_cast<int>(key & 0xFFFFFFFF);
  return std::make_pair(device_id, cu_count);
}

static ObjPool<hipStream_t, ServiceStreamKey> service_stream_pool;

thread_local static ObjCache<hipStream_t, ServiceStreamKey> service_stream_cache(
  [](const ObjCache<hipStream_t, ServiceStreamKey>::Data &d) { service_stream_pool.store(d); }
);

struct ServiceStreamCtl {
  hipStream_t stream;
  hipEvent_t start_event;
  hipEvent_t end_event;
};


bool get_service_stream(int math_sm_count, hipStream_t stream, struct ServiceStreamCtl &ctl)
{
  if (math_sm_count == 0)
    return false; // No service stream needed

  int device_id;
  int device_cu_count = 0;
  NVTE_CHECK_CUDA(hipGetDevice(&device_id));
  NVTE_CHECK_CUDA(hipDeviceGetAttribute(&device_cu_count, hipDeviceAttributeMultiprocessorCount, device_id));
  if (math_sm_count < 0 || math_sm_count > device_cu_count)
  {
    std::cerr << "[WARNING] Invalid math_sm_count: " << math_sm_count << std::endl;
    return false; // Invalid math_sm_count
  }
  else if (math_sm_count == device_cu_count)
  {
    return false; // math_sm_count == device_cu_count is equivalent to math_sm_count == 0
  }

  // Check if stream is capturing
  hipStreamCaptureStatus captureStatus;
  NVTE_CHECK_CUDA(hipStreamIsCapturing(stream, &captureStatus));
  if (captureStatus != hipStreamCaptureStatusNone)
  {
    std::cerr << "[WARNING] Cannot use math_sm_count with captured stream" << std::endl;
    return false; // Cannot use service stream with captured stream
  }

  ServiceStreamKey key = make_service_stream_key(device_id, math_sm_count);
  CacheEntry<hipStream_t> streamEntry = service_stream_cache.get(key, stream);
  if (!streamEntry.isValid()) {
    /* There is no entry in the cache, try the following:
      * 1. Try to acquire any available stream form the cache.
      * 2. If not available, try to acquire any available stream form the pool.
      * 3. If still not available, create a new stream and event. */
    bool b_log = false;
    if (const char* env_p = std::getenv("NVTE_LOG_MATH_SM_COUNT") ) {
      b_log = (env_p != nullptr) && (std::string(env_p) == "1");
    }
    streamEntry = service_stream_cache.acquire(key, stream);
    if (!streamEntry.isValid()) {
      streamEntry = service_stream_pool.acquire(key, stream);
    }
    if (!streamEntry.isValid())
    {
      const uint32_t maskSize = (math_sm_count + 31) / 32;
      std::vector<uint32_t> mask(maskSize, (uint32_t)-1);
      if (math_sm_count % 32 != 0)
      {
        mask[maskSize-1] = (1UL << (math_sm_count % 32)) - 1;
      }
      NVTE_CHECK_CUDA(hipExtStreamCreateWithCUMask(&streamEntry.value, maskSize, mask.data()));
      NVTE_CHECK_CUDA(hipEventCreateWithFlags(&streamEntry.event, hipEventDisableTiming));
      if (b_log)
      {
        std::cout << "[DEBUG] Created service stream for device " << device_id
                  << " with " << math_sm_count << " CUs" << std::endl;
      }
    }
    else if (b_log)
    {
      std::cout << "[DEBUG] Reusing service stream for device " << device_id
                << " with " << math_sm_count << " CUs" << std::endl;
    }
    service_stream_cache.set(key, stream, streamEntry);
  }

  ctl.stream = streamEntry.value;
  ctl.end_event = streamEntry.event;
  NVTE_CHECK_CUDA(hipEventCreateWithFlags(&ctl.start_event, hipEventDisableTiming));
  NVTE_CHECK_CUDA(hipEventRecord(ctl.start_event, stream));
  NVTE_CHECK_CUDA(hipStreamWaitEvent(ctl.stream, ctl.start_event, 0));
  return true; 
}

void release_service_stream(hipStream_t stream, struct ServiceStreamCtl &ctl)
{
    NVTE_CHECK_CUDA(hipEventRecord(ctl.end_event, ctl.stream));
    NVTE_CHECK_CUDA(hipStreamWaitEvent(stream, ctl.end_event, 0));
    //TODO: when event are really destroyed (documentation says on devide synchronize) and how much overhead is to create them
    //May need to store event in eventPool and reuse them after thy are recorded
    NVTE_CHECK_CUDA(hipEventDestroy(ctl.start_event));
}

} // namespace


void cublas_gemm(const Tensor *inputA, const Tensor *inputB, Tensor *outputD,
                 const Tensor *inputBias, Tensor *outputPreGelu, int m, int n, int k, int lda,
                 int ldb, int ldd, bool transa, bool transb, bool grad,
                 void *workspace, size_t workspaceSize, bool accumulate, bool use_split_accumulator,
                 int math_sm_count, int m_split, int n_split, bool gemm_producer,
                 const Tensor *inputCounter, hipStream_t stream, int compute_stream_offset)
{
  ServiceStreamCtl ss_ctl;
  bool use_service_stream =
      (math_sm_count != 0) ? get_service_stream(math_sm_count, stream, ss_ctl) : false;

  NVTE_CHECK(compute_stream_offset >= -1 && compute_stream_offset < num_streams);

  hipblasLtHandle_t handle = nullptr;
  if (compute_stream_offset != -1) {
    // Init hipblaslt handles (once, globally)
    static std::once_flag init_flag;
    static hipblasLtHandle_t hipblaslt_handles[num_streams];
    std::call_once(init_flag, init_hipblaslt_handles, hipblaslt_handles);

    handle = hipblaslt_handles[compute_stream_offset];
  }

  hipblaslt_gemm(inputA, inputB, outputD, inputBias, outputPreGelu, 
                  m, n, k, lda, ldb, ldd,
                  (transa) ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                  (transb) ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                  grad,
                  workspace, workspaceSize, accumulate, use_split_accumulator,
                  math_sm_count, m_split, n_split, gemm_producer,
                  inputCounter, use_service_stream ? ss_ctl.stream : stream, handle);

  if (use_service_stream)
  {
    release_service_stream(stream, ss_ctl);
  }
}

} //namespace transformer_engine
