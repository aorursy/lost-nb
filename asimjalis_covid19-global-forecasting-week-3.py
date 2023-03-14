#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('javascript', '', "// Build table of contents.\n$.getScript('https://kmahelona.github.io/ipython_notebook_goodies/ipython_notebook_toc.js')\n// Disable output area scrolling.\n// IPython.OutputArea.auto_scroll_threshold = 1")


# In[2]:


get_ipython().system('pip install hy > /dev/null')


# In[3]:


# Hy Magic
import IPython
def hy_eval(*args):
    import hy
    try: return hy.eval(hy.read_str("(do\n"+"".join(map(lambda s:s or "",args))+"\n)\n"),globals())
    except Exception as e: print("ERROR:", str(e)); raise e
@IPython.core.magic.register_line_cell_magic
def h(*args): return hy_eval(*args) # Prints result useful for debugging.
@IPython.core.magic.register_line_cell_magic
def hs(*args):hy_eval(*args) # Silent. Does not print result.
del h, hs


# In[4]:


get_ipython().run_cell_magic('hs', '', '(import  [useful [*]])\n(require [useful [*]])')


# In[5]:


get_ipython().run_cell_magic('h', '', '\n; COVID19 Global Forecasting (Week 3)\n\n; vim: filetype=lisp tw=9999 nowrap\n\n; [Paths]\n\n(=> cor-prefix (-> "covid19-global-forecasting-week-3" kag-comp->prefix))\n(=> cor-train-csv (+ cor-prefix "train.csv"))\n(=> cor-test-csv (+ cor-prefix "test.csv"))\n(=> cor-submission-csv (+ cor-prefix "submission.csv"))\n\n; [Macros & Utilities]\n\n(import [seaborn :as sns])\n(defmacro symbol-to-string [sym] `(mangle (name ~sym)))\n(defmacro pd-define [df dst &rest forms]\n  `(.assign ~df #** {~(symbol-to-string dst) (fn [$] ~@forms)}))\n(defmacro pd-filter [df &rest forms]\n    `(-> ~df (.where (fn [$] ~@forms)) (.dropna)))\n(defmacro pd-define-plot [df dst &rest forms]\n  `(-> ~df (pd-define ~dst ~@forms) (pd-plot [~(symbol-to-string dst)])))\n(defn pd-regression [df x y]\n  (setv x (-> df (get x) (.to-numpy)))\n  (setv y (-> df (get y) (.to-numpy)))\n  (setv line (stats.linregress x y))\n  (print :sep "\\n"\n    f"slope={line.slope}"\n    f"intercept={line.intercept}"\n    f"pvalue={line.pvalue}"\n    f"rvalue={line.rvalue}"\n    f"stderr={line.stderr}")\n  (plt.figure)\n  (plt.plot x y "o" :label "original data")\n  (plt.plot x (+ line.intercept (* line.slope x)) "r" :label "fitted line")\n  (plt.legend)\n  (plt.show)\n  (plt.close)\n  df)\n\n(defmacro pd-fork [df &rest forms]\n  `(do (setv $ ~df) ~@forms $))\n\n(defn cor-csv->df [file-csv id]\n  (-> file-csv\n    (pd.read-csv :dtype {id object})\n    (.fillna "") \n    ; Clean-up regions\n    (pd-define RegionId (-> ($.Country_Region.str.cat :sep ":" $.Province_State)))\n    (pd-drop ["Country_Region" "Province_State"])\n    (pd-date-string-to-date "Date" "Date")\n    (pd-date-to-std-day "t" "Date") \n    (.set-index "Date" :drop False)\n    (pd-rename {"ConfirmedCases" "y"})\n    (pd-rename {"Fatalities"     "z"})\n  ))\n\n; Use this to remove trailing colon\n;=> (-> cor-train-csv (pd.read-csv) (.fillna "") (pd-assign region (-> ($.Country_Region.str.cat :sep ":" $.Province_State) (. str) (.replace ":$" ""))) (.query "region==\'US:California\'")\n\n(defn kag-log [message]\n  (=> timestamp (-> (datetime.now) (str)) )\n  (print f"[{timestamp}] {message}"))\n\n; [Models]\n\n(defn sigmoid [x x0 r] \n  (-> x (- x0) (* (- r)) (np.exp) (+ 1) (np.reciprocal)))\n(defn n-sigmoid [x N x0 r] \n  (-> x (- x0) (* (- r)) (np.exp) (+ 1) (np.reciprocal) (* N)))\n(defn log-n-sigmoid [x N x0 r] \n  (-> x (- x0) (* (- r)) (np.exp) (+ 1) (np.reciprocal) (* N) (np.log1p)))\n(defn quadratic [x a b c] \n  (-> a (* x) (+ b) (* x) (+ c)))\n\n(import pylab)\n(import [scipy.optimize [curve-fit]])\n\n(defn const-func [c] (fn [t] c))\n(defn const-model [c] (kw->obj :func (const-func c) :popt (np.array [0 0 0]) :mape 0.0))\n\n(defn pd-curve-model [df y-col x-col f p0 &optional [plot False]]\n  (=> df-clean (-> df (.query f"{y-col} > 0.0")))\n  (=> y-max (-> df (get y-col) (.max)))\n  (if (-> df-clean (len) (<= 4)) (return (const-model y-max)))\n  (if (-> df-clean (len) (> 4)) (=> df df-clean))\n  (=> x (-> df (get x-col)))\n  (=> y (-> df (get y-col)))\n  (=> (, popt pcov) (curve-fit f x y :p0 p0 :maxfev 20000))\n  (=> func (fn [x] (f x #* popt)))\n  (=> y-hat (func x))\n  (=> mape (-> (np.abs y) (- (np.abs y-hat)) (np.abs) (np.mean) (/ (np.abs (np.mean y)))))\n  (if plot \n    (do\n      (pylab.plot x y "o" :label "data")\n      (pylab.plot x y-hat :label "fit")\n      (pylab.ylim (np.min y) (np.max y))\n      (pylab.legend :loc "best")\n      (pylab.show)))\n  (kw->obj :func func :popt popt :mape mape))\n\n; [Testing & Submission]\n\n(=> cor-regions (-> (cor-csv->df cor-test-csv  "ForecastId") (get "RegionId") (.unique) (list)))\n(=> cor-y-models {})\n(=> cor-z-models {})\n\n(defn cor-region->train-df [region]\n  (-> cor-train-csv\n    (cor-csv->df "Id") \n    (.query "RegionId == @region")\n  ))\n\n(defn cor-region->test-df [region]\n  (-> cor-test-csv\n    (cor-csv->df "ForecastId") \n    (.query "RegionId == @region")\n  ))\n\n(defn cor-train-df->model [df y-col x-col &optional [plot False]]\n  (=> y (-> df (get y-col)))\n  (=> y-max (-> y (.max)))\n  (=> n-estimate (-> y-max (/ 2))) \n  (=> p0 [n-estimate 87 0.2])\n  (=> model (pd-curve-model df y-col x-col n-sigmoid :p0 p0 :plot plot)))\n\n(import math)\n(defn is-num-bad [x] (or (-> x (math.isfinite) (not)) (-> x (> 0.5))))\n(defn cor-check-bad-mape [mape] (if (is-num-bad mape) (print f"--> BAD mape={mape}")))\n\n(defn cor-build-models [&optional [exclude-list []]]\n  ; Build models per region\n  (global cor-y-models cor-z-models)\n  (=> executions [])\n  (=> exclude-set (set exclude-list))\n  (for [region cor-regions] \n    (print f"train: region={region}")\n    (=> train-df (cor-region->train-df region))\n    (for [series ["y" "z"]]\n      (if (and (= series "z") (-> region (in exclude-set))) (continue))\n      (=> model  (cor-train-df->model train-df series "t"))\n      (cor-check-bad-mape model.mape)\n      (if (= series "y") (-> cor-y-models (setf region model)))\n      (if (= series "z") (-> cor-z-models (setf region model)))\n      (executions.append \n        { "region" region \n          "series" series \n          "mape" model.mape\n          "popt" model.popt \n          "N"    (get model.popt 0)\n          "x0"   (get model.popt 1)\n          "r"    (get model.popt 2)\n          "bad_mape" (is-num-bad model.mape) })))\n  executions)\n\n; [Manual]\n\n(defn chk-make-executions []\n  (=> executions (cor-build-models)))\n\n(defn chk-brunei-fit-1 []\n  (-> "Brunei:" (cor-region->train-df) (cor-train-df->model "y" "t" :plot True)))\n\n(defn chk-brunei-fit-2 []\n  (-> "Brunei:" \n    (cor-region->train-df) \n    (pd-curve-model "y" "t" n-sigmoid :p0 [40 87 0.2] :plot True)))\n\n; [Build Models]\n\n(=> executions (cor-build-models))\n(-> executions \n  (pd.DataFrame)\n  (.sort-values :by "mape" :ascending False) (display))\n\n; [Submission]\n\n(defmacro pd-assign [df dst &rest forms]\n  `(.assign ~df #** {~(name dst) (fn [$] ~@forms)}))\n\n(defn cor-prepare-submission []\n  (=> submission-df (pd.DataFrame))\n  (for [region cor-regions]\n    (print f"test: region={region}")\n    (=> y-model (-> cor-y-models (get region)))\n    (=> z-model (-> cor-z-models (get region)))\n    (-> (cor-region->test-df region)\n      (pd-assign ConfirmedCases (-> $.t (y-model.func) (np.round :decimals 1)))\n      (pd-assign Fatalities     (-> $.t (z-model.func) (np.round :decimals 1)))\n      (pd-save predict-df))\n    ; Append predict-df to submission-df \n    (=> submission-df (pd.concat [submission-df predict-df])))\n  submission-df)\n\n(=> submission-df (cor-prepare-submission))\n\n(pd.set-option "display.float_format" (fn [x] (% "%.2f" x)))\n\n(-> submission-df \n  (pd-keep ["ForecastId" "ConfirmedCases" "Fatalities"] ) \n  (.to-csv "submission.csv" :index False :float-format "%.1f"))\n')

