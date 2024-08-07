def visual(adata,df_label):
    import matplotlib as mpl
    import scanpy as sc
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import warnings
    from anndata import AnnData
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams["font.sans-serif"] = "Arial"
    warnings.filterwarnings('ignore')

    sc.settings.set_figure_params(dpi=80,dpi_save=600,facecolor='white')
    adata.var_names_make_unique()
    adata
    adata.obsm['spatial']
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    print(f"#cells after MT filter: {adata.n_obs}")
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)
    sc.pp.pca(adata,n_comps=10)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, key_added="clusters")
    
    numpy_array = df_label.to_numpy()
    numpy_array
    unlisted_arr = numpy_array.flatten()
    dat=unlisted_arr
    
    cc=adata.obs.clusters
    cc.index
    obj = pd.Series(dat)
    obj.index=cc.index
    obj=obj.astype('category')
    adata.obs.clusters=obj
    sc.pl.spatial(adata, img_key="hires", color="clusters", size=1.5)
    sc.pl.spatial(adata, img_key="hires", color="clusters", size=1.5,save="visualdomainplot_plot.pdf")