#!/usr/bin/env python
# coding: utf-8



get_ipython().run_cell_magic('javascript', '', "/* Build table of contents. */\n$.getScript('https://kmahelona.github.io/ipython_notebook_goodies/ipython_notebook_toc.js')\n/* Disable output area scrolling. */\nIPython.OutputArea.auto_scroll_threshold = 9999")




get_ipython().system('pip install hy > /dev/null')




# Hy Magic
import IPython
def hy_eval(*args):
    import hy
    try:
        return hy.eval(hy.read_str("(do\n"+"".join(map(lambda s:s or "",args))+"\n)\n"),globals())
    except Exception as e:
        print("ERROR:", str(e))
        raise e
@IPython.core.magic.register_line_cell_magic
def h(*args): return hy_eval(*args) # Prints result useful for debugging.
@IPython.core.magic.register_line_cell_magic
def hs(*args):hy_eval(*args) # Silent. Does not print result.
del h, hs




get_ipython().run_cell_magic('hs', '', '(import  [useful [*]])\n(require [useful [*]])')




get_ipython().run_cell_magic('hs', '', '\n; Define paths\n(=> corona-prefix (-> "covid19-global-forecasting-week-2" kag-comp->prefix))\n(=> corona-train-csv (+ corona-prefix "train.csv"))\n(=> corona-test-csv (+ corona-prefix "test.csv"))\n(=> corona-submission-csv (+ corona-prefix "submission.csv"))\n\n; Read CSV, create distinct regions\n(defn corona-csv->df [file-csv id]\n  (-> file-csv\n    (pd.read-csv :dtype {id object})\n    (.fillna "")\n    (pd-assign-> "RegionId" (-> ($.Country_Region.str.cat :sep ":" $.Province_State)))))\n\n(defn corona-train-csv->df [] (corona-csv->df corona-train-csv "Id"))\n(defn corona-test-csv->df  [] (corona-csv->df corona-test-csv  "ForecastId"))\n\n; Get all regions\n(defn corona-train->regions [] (-> (corona-train-csv->df) (.RegionId.unique) (list)))\n(defn corona-test->regions  [] (-> (corona-test-csv->df)  (.RegionId.unique) (list)))\n\n; Read CSV, prepare DF\n(defn corona-train-region->df [region-id]\n  (-> (corona-train-csv->df)\n\n    ; Filter by region\n    (pd-filter-> (= $.RegionId region_id))\n   \n    ; Standardize column names\n    (pd-rename {"Date" "t" "ConfirmedCases" "Y" "Fatalities" "Z"})\n    (pd-keep   ["t" "Y" "Z"])\n    \n    ; Standardize dates to start on Jan 1, 2020\n    (pd-date-string-to-date "t" "t")\n    (.set-index "t" :drop False)\n    (pd-date-to-std-day "t" "t")))\n\n(defn corona-train-country-state->df [country state]\n  (corona-train-region->df (+ country ":" state)))\n\n; Read CSV, prepare DF\n(defn corona-test-region->df [region-id]\n  (-> (corona-test-csv->df)\n\n    ; Filter by region\n    (pd-filter-> (= $.RegionId region_id))\n   \n    ; Standardize column names\n    (pd-rename {"Date" "t"})\n\n    ; Standardize dates to start on Jan 1, 2020\n    (pd-date-string-to-date "t" "t")\n    (.set-index "t" :drop False)\n    (pd-date-to-std-day "t" "t")\n\n    ; Keep essentials columns\n    (pd-keep   ["ForecastId" "t"])))')




get_ipython().run_cell_magic('hs', '', '(import [math [log exp]])\n\n(pd.set-option "display.float_format" (fn [x] (% "%.2f" x)))\n(-> (corona-train-country-state->df "US" "California")\n \n  ; Drop zero days\n  (pd-filter-> (> $.Y 0.0))\n  \n  ; Calculate Y_prime\n  (pd-assign-> "Y_prime" (-> $.Y (np.gradient :edge-order 1 $.t)))\n  \n  ; Calculate Y_prime_prime\n  (pd-assign-> "Y_prime_prime" (-> $.Y_prime (np.gradient :edge-order 1 $.t)))\n \n  ; Calculate Y_prime/Y\n  (pd-assign-> "Y_prime_over_Y" (/ $.Y_prime $.Y))\n\n  ; Calculate log_y_prime which equals Y_prime/Y\n  (pd-assign-> "log_Y" (np.log $.Y))\n  (pd-assign-> "log_Y_prime"   (-> $.log_Y (np.gradient :edge-order 1 $.t)))\n  (pd-assign-> "log_Y_prime_smooth" (-> $.log_Y_prime (.ewm :alpha 0.25) (.mean)))\n  (pd-plot ["log_Y_prime" "log_Y_prime_smooth"])\n  (pd-plot ["log_Y_prime_smooth"])\n  (pd-assign-> "Y_prime_over_Y" $.log_Y_prime_smooth)\n\n  (pd-add-regression "log_Y_m" "log_Y_b" "log_Y_pvalue" "log_Y_rvalue" "t" "log_Y")\n  (pd-assign-> "r1" $.log_Y_m)\n  (pd-assign-> "N1" (/ (* $.r1 $.Y $.Y) (- (* $.r1 $.Y) $.Y_prime)))\n\n  (pd-add-regression "m" "b" "pvalue" "rvalue" "Y" "Y_prime_over_Y")\n \n  (pd-assign-> "r" $.b)\n  (pd-assign-> "N" (-> (- $.b) (/ $.m)))\n \n \n  (pd-assign-> "t_infl"  (-> $.N (/ $.Y) (- 1) (np.log) (-) (/ $.r) (-) (+ $.t)))\n  (pd-assign-> "t_infl_date"  (-> $.t_infl (ps-std-day-to-date)))\n \n  ; Growth rate\n  (pd-assign-> "growth" (/ $.Y ($.Y.shift)))\n  (pd-assign-> "doubling" (/ (np.log 2) (- (np.log $.Y) (np.log ($.Y.shift)))))\n \n  ; Model\n  (pd-assign-> "Y_predict" \n    (/ (np.mean $.N) \n       (-> $.t (- (np.mean $.t_infl)) (* (np.mean $.r)) (* -1) (np.exp) (+ 1))))\n  (pd-save kag-work)\n)\n\n(-> kag-work\n  (pd-plot ["Y_prime_prime"])\n  (pd-plot ["log_Y"])\n  (pd-regression "t" "log_Y")\n  (pd-plot ["N1"])\n  (pd-plot ["log_Y_prime"])\n  (pd-plot ["Y_prime_over_Y" "log_Y_prime"])\n  (pd-plot ["Y_prime_over_Y"] :index "Y")\n  (pd-regression "Y" "Y_prime_over_Y")\n  (pd-plot ["growth" "doubling"])\n  ;(pd-plot ["Y" "Y_prime"])  (pd-fork (pd-keep ["Y" "Y_prime"]) (display))\n  (pd-describe)\n  (display)\n)')




get_ipython().run_cell_magic('h', '', '(import [datetime [datetime timedelta]])\n\n(defn corona-train-df->model-log-regression [df y t]\n  ; Drop zeroes\n  (=> df (-> df (pd-filter-> (-> $ (get y) (> 0.0)))))\n  (=> y (-> df (get y) ))\n  (=> t (-> df (get t) ))\n  (=> y_log       (-> y (np.log)))\n  (=> line        (stats.linregress t y_log))\n  (=> m           (-> line.slope (np.float64)))\n  (=> b           (-> line.intercept (np.float64)))\n  (=> r           (-> line.slope (np.float64)))\n  (=> func        (fn [t] (-> t (* m) (+ b) (np.round :decimals 1) (np.exp))))\n  (locals->obj ["LinregressResult" "float64" "datetime" "function"]))\n\n(defn corona-train-df->model-sigmoid-smoothing [df y t]\n  (-> df\n    (pd-filter-> (-> (get $ y) (> 0.0)))\n    (pd-assign-> "y_prime" (-> (get $ y) (np.gradient :edge-order 1 $.t)))\n    (pd-assign-> "y_prime_over_y" (-> $.y_prime (/ (get $ y))))\n    (pd-assign-> "y_log_prime" (-> (get $ y) (np.log1p) (np.gradient :edge-order 1 $.t)))\n    (pd-assign-> "y_log_prime" (-> $.y_log_prime (.ewm :alpha 0.25) (.mean)))\n    (pd-save df))\n  (=> y           (-> df (get y) ))\n  (=> t           (-> df (get t) ))\n  (=> y_prime     (-> df (get "y_prime") ))\n  (=> y_log_prime (-> df (get "y_log_prime") ))\n  (=> line        (stats.linregress y y_log_prime))\n  (=> r           (-> line.intercept (np.float64)))\n  (=> N           (-> (- r) (/ line.slope) (np.float64)))\n  (=> t_infl      (-> N (/ y) (- 1) (np.log1p) (-) (/ r) (-) (+ t) (np-dropna) (np.mean)))\n  (=> t_infl_date (-> t_infl (timedelta) (+ std-day-0)))\n  (=> y_shift     (-> y (.shift)))\n  (=> growth      (-> y (/ (y.shift)) (np.mean)))\n  (=> doubling    (-> (np.log 2) (/ (np.log growth))))\n  (=> func        (fn [t] (np.round :decimals 1 (/ N (-> t (- t_infl) (* r) (* -1) (np.expm1) (+ 1))))))\n  (locals->obj ["LinregressResult" "float64" "datetime" "function"]))\n\n(defn corona-fixed-func [value] (fn [t] value))\n\n(defn np-last-2 [ser] \n  (-> ser (get (cut ser.index -2)) (tuple)))\n\n(defn np-is-number [ser] \n  (~ (| (np.isnan ser) (np.isinf ser))))\n\n(defn np-drop-nan-inf [ser]\n  (-> ser (np-is-number) (np.extract ser)))\n\n(defn corona-train-df->model-sigmoid [df y-col t]\n  ; Drop zeroes\n  (=> df (-> df (pd-filter-> (-> $ (get y-col) (> 0.0)))))\n  (=> y (-> df (get y-col) ))\n  (=> t (-> df (get t) ))\n  (if (-> df (len) (= 0)) (return (dict->obj {"func" (corona-fixed-func 0.0)})))\n  (=> y_max (-> y (.max)))\n  (if (-> df (len) (< 4)) (return (dict->obj {"func" (corona-fixed-func y_max)})))\n  (=> [y-last1 y-last0] (-> y np-last-2))\n  (if (= y-last0 y-last1) (return (dict->obj {"func" (corona-fixed-func y-last0)})))\n  (if (= y-last1 0)       (return (dict->obj {"func" (corona-fixed-func y-last0)})))\n  (=> y-last-chg (-> y-last0 (- y-last1) (/ y-last1)))\n  (if (< y-last-chg 0.06) (return (dict->obj {"func" (corona-fixed-func y-last0)})))\n  (=> y_prime     (-> y (np.gradient :edge-order 1 t)))\n  (=> y_prime_over_y (-> y_prime (/ y)))\n  (=> y_log_prime (-> y (np.log1p) (np.gradient :edge-order 1 t)))\n  (=> line        (stats.linregress y y_log_prime))\n  (=> r           (-> line.intercept (np.float64)))\n  (=> N           (-> (- r) (/ line.slope) (np.float64)))\n  (if (or (< N 1) (np.isnan N)) (=> N 1))\n  (=> t_infl      (-> N (/ y) (- 1) (np.log1p) (-) (/ r) (-) (+ t) (np-drop-nan-inf) (np.mean)))\n  (if (np.isnan t_infl) (=> t_infl 100))\n  (=> t_infl_date (-> t_infl (timedelta) (+ std-day-0)))\n  (=> y_shift     (-> y (.shift)))\n  (=> growth      (-> y (/ (y.shift)) (np.mean)))\n  (=> doubling    (-> (np.log 2) (/ (np.log growth))))\n  (=> func        \n    (fn [t] \n      (=> y_predict (np.round :decimals 1 (/ N (-> t (- t_infl) (* r) (* -1) (np.expm1) (+ 1)))))\n      (=> y_predict (-> y_predict (np.nan-to-num :posinf y_max :neginf 0)))\n      y_predict))\n  (locals->obj ["LinregressResult" "float64" "datetime" "function"]))')




get_ipython().run_cell_magic('h', '', '(=> corona-train-df->model corona-train-df->model-sigmoid)\n\n(=> model \n  (-> (corona-train-country-state->df "Anhui" "China")\n    ; Drop zero case days\n    (pd-filter-> (> $.Y 0.0))\n   \n    ; Create model\n    (corona-train-df->model "Z" "t")\n  ))\n\n(p model)\n(model.func 0)')




get_ipython().run_cell_magic('hs', '', '\n(=> corona-y-models {})\n(=> corona-z-models {})\n\n(defn kag-log [message]\n  (=> timestamp (-> (datetime.now) (str)) )\n  (print f"[{timestamp}] {message}"))\n\n(defn corona-build-models []\n  ; Build models per region\n  (=> train-regions (corona-train->regions))\n  (for [region train-regions] \n    (=> train-df (corona-train-region->df region))\n    (=> y-model (corona-train-df->model train-df "Y" "t"))\n    (=> z-model (corona-train-df->model train-df "Z" "t"))\n    (-> corona-y-models (setf region y-model))\n    (-> corona-z-models (setf region z-model))))\n(corona-build-models)')




get_ipython().run_cell_magic('h', '', '\n(defn corona-prepare-submission []\n  (=> submission-df (pd.DataFrame))\n  (=> test-regions (corona-test->regions))\n  (for [region test-regions]\n    (=> y-model (-> corona-y-models (get region)))\n    (=> z-model (-> corona-z-models (get region)))\n   \n    (-> (corona-test-region->df region)\n      (pd-assign-> "Y_predict" (-> $.t (y-model.func)))\n      (pd-assign-> "Z_predict" (-> $.t (z-model.func)))\n      (pd-save predict-df))\n    ; Append predict-df to submission-df \n    (=> submission-df (pd.concat [submission-df predict-df])))\n  submission-df)\n  \n(=> submission-df (corona-prepare-submission))\n(-> submission-df \n  (pd-rename {"Y_predict" "ConfirmedCases" "Z_predict" "Fatalities"})\n  (pd-keep ["ForecastId" "ConfirmedCases" "Fatalities"] ) \n  (.to-csv :index False "submission.csv"))')




get_ipython().run_cell_magic('hs', '', '\n; Cross-Validation\n(=> corona-train-df->model corona-train-df->model-sigmoid)\n\n; Train test split\n(=> full-df (corona-train-country-state->df "China" "Fujian"))\n(=> train-test-split 0.95)\n(=> split-point (-> full-df (len) (* train-test-split) (int)))\n(=> (, train-df test-df) (np.split full-df [split-point]))\n\n; Build model\n(=> y-model (corona-train-df->model train-df "Y" "t"))\n(=> z-model (corona-train-df->model train-df "Z" "t"))\n;(display [y-model z-model])\n\n; Test model predictions\n(-> test-df \n (pd-assign-> "Y_predict" (-> $.t (y-model.func)))\n (pd-assign-> "Z_predict" (-> $.t (z-model.func)))\n (pd-save predict-df)\n)\n \n; Compute error\n(=> y-error (-> predict-df (pd-prediction->error "Y_predict" "Y")))\n(=> z-error (-> predict-df (pd-prediction->error "Z_predict" "Z")))\n\n(defn pd-plot-predict-vs-actual [df x y-model y-actual]\n  (-> df\n    (.set-index x)\n    (pd-plot [y-model y-actual]))\n  df)\n\n(-> predict-df \n (pd-plot-predict-vs-actual "t" "Y" "Y_predict")\n (pd-plot-predict-vs-actual "t" "Z" "Z_predict"))\n\n(display [y-error.rmsle z-error.rmsle])')

