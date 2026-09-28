"""Import every compiled module the pipeline needs and run a tiny smoke computation."""
import sys
print("PY", sys.version, sys.executable)
import numpy, pandas, scipy, scipy.sparse, scipy.spatial, scipy.stats
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm, rapidfuzz
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
from sparse_dot_topn import sp_matmul_topn
import pyarrow
X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3)).fit_transform(["acme robotics", "acme robotic inc", "zen traders"])
C = sp_matmul_topn(X, X.T.tocsr(), top_n=2)
print("topn nnz", C.nnz, "cpdist", cpdist(["abc"], ["abd"], scorer=fuzz.ratio))
import numpy as np
b = lightgbm.train({"objective": "binary", "verbose": -1}, lightgbm.Dataset(np.random.rand(200, 3), label=np.random.randint(0, 2, 200)), 5)
print("versions numpy", numpy.__version__, "pandas", pandas.__version__, "scipy", scipy.__version__,
      "sklearn", sklearn.__version__, "lightgbm", lightgbm.__version__, "rapidfuzz", rapidfuzz.__version__,
      "pyarrow", pyarrow.__version__)
print("ENV OK")
