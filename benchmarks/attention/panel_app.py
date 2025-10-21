import pandas as pd
import panel as pn
import seaborn as sns
from matplotlib.figure import Figure
pn.extension(design="material", sizing_mode="stretch_width")

@pn.cache
def get_data():
  return pd.read_csv("output_main.csv")
df=get_data().fillna("NaN")
VARIABLES = ["dtype", "seq_desc_format"]
def _selector_widgets():
    for cat in ("attn_bias_type", "attn_mask_type", "qkv_layout", "bias_shape", "is_training","swa", "dropout"):
        yield pn.widgets.Select(name=cat, options=list(df[cat].unique()))


selector_widgets = list(_selector_widgets())

def make_plot(attn_bias_type, attn_mask_type, qkv_layout, bias_shape, is_training, swa, dropout):
    fig = Figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    
    subset = df[
        (
            (df["attn_bias_type"]==attn_bias_type) &
            (df["attn_mask_type"]==attn_mask_type) &
            (df["qkv_layout"]==qkv_layout) &
            (df["bias_shape"]==bias_shape) &
            (df["is_training"]==is_training) &
            (df["swa"]==swa) &
            (df["dropout"]==dropout)
        )
    ]
    subset = subset[subset.time < subset.time.quantile(.95)]
    subset = subset[subset.time > subset.time.quantile(.05)]
    if not subset.empty:
        sns.swarmplot(ax=ax, data=subset,x="seq_desc_format", y="time", hue="dtype", dodge=True)
        return fig

bound_make_plot = pn.bind(make_plot, 
        attn_bias_type=selector_widgets[0],
        attn_mask_type=selector_widgets[1],
        qkv_layout=selector_widgets[2],
        bias_shape=selector_widgets[3],
        is_training=selector_widgets[4],
        swa=selector_widgets[5],
        dropout=selector_widgets[6],
)

template = pn.template.BootstrapTemplate(
    title='JAX Fused Attention Benchmarks',
    sidebar=selector_widgets,
)
template.main.append(pn.pane.Matplotlib(bound_make_plot, dpi=144, height=500))
template.servable();