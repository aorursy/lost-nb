#!/usr/bin/env python
# coding: utf-8



get_ipython().system('pip install hy > /dev/null')




# Hy Magic
import IPython
def hy_eval(*args):import hy;return hy.eval(hy.read_str("(do\n"+"".join(map(lambda s:s or "",args))+"\n)\n"),globals())
@IPython.core.magic.register_line_cell_magic
def h(*args):hy_eval(*args) # Silent: Does not print result.
@IPython.core.magic.register_line_cell_magic
def hh(*args): return hy_eval(*args) # Verbose: Prints result.
del h, hh




get_ipython().run_cell_magic('h', '', '(import  [useful [*]])\n(require [useful [*]])')




get_ipython().run_cell_magic('h', '', '; Figure out location of data\n(s covid-root-kaggle "/kaggle/input")\n(s covid-root-laptop "$HOME/d")\n(s covid-root \n   (-> "/kaggle" \n       (os.path.exists) \n       (if covid-root-kaggle covid-root-laptop)\n       (os.path.expandvars)))\n(s covid-prefix (+ covid-root "/covid19-local-us-ca-forecasting-week-1/ca_"))\n(s covid-train  (+ covid-prefix "train.csv"))\n(s covid-test   (+ covid-prefix "test.csv"))\n(s covid-submit (+ covid-prefix "submission.csv"))')




get_ipython().run_cell_magic('hh', '', '; Find out day of week\n(-> covid-train\n  (pd.read-csv) \n  (pd-keep ["Date" "ConfirmedCases" "Fatalities"]) \n  (.assign :Date (fn [x] (pd.to-datetime :yearfirst True x.Date)))\n  (.assign :Day  (fn [x] (-> (x.Date.dt.day_name))))\n  (.head)\n)')




get_ipython().run_cell_magic('h', '', '; Plot the data\n\n(import math)\n(-> 39.94 (* 1000) (* 1000) (math.log1p) (s1 ca-pop-log))\n\n(-> covid-train\n  (pd.read-csv) \n  (pd-keep ["Date" "ConfirmedCases" "Fatalities"]) \n  (.where (fn1-> (. ConfirmedCases) (> 0.0))) (.dropna)\n  (.assign :ConfirmedCases (fn1-> (. ConfirmedCases) (np.log1p))) \n  (.assign :Fatalities     (fn1-> (. Fatalities)     (np.log1p)))\n  (.assign :Date           (fn1-> (. Date)           (pd.to-datetime :yearfirst True)))\n  (.assign :DayOfYear      (fn1-> (. Date) (. dt) (. dayofyear)))\n  (.assign :Day            (fn1-> (. Date) (. dt) ( .day_name))) \n  (.dropna)\n  (s1 df1))\n\n(-> df1\n  (pd-keep ["Date" "ConfirmedCases" "Fatalities"])\n  (.set-index "Date")\n  (pd-plot "Log1p")\n  (display))')




get_ipython().run_cell_magic('h', '', '; Define data frames.\n\n(s MILLION    (-> 1000 (* 1000)))\n(s population (-> 39.94 (* MILLION)))\n\n(-> covid-train\n  (pd.read-csv) \n  (pd-keep ["Date" "ConfirmedCases" "Fatalities"]) \n  (.rename :columns {"ConfirmedCases" "Conf" "Fatalities" "Dead" })\n  (.assign :Conf (fn1-> (. Conf) (/ population))) \n  (.assign :Dead (fn1-> (. Dead) (/ population)))\n  (.dropna)\n  (s1 df))\n\n(setv conf-actual (df.Conf.rename "Actual"))\n(setv dead-actual (df.Dead.rename "Actual"))')




get_ipython().run_cell_magic('h', '', '(import [scipy.optimize [minimize]])\n(import [math [exp]])\n\n; Define conf model then run it.\n\n(defn conf-model [a alpha t0 t]\n  (setv t-delta (- t t0))\n  (if (< t-delta 0)\n    0.0\n    (** (- 1 (exp (* (- a) t-delta))) alpha)))\n\n(defn conf-model-loss [x df]\n  (setv (, a alpha t0) x)\n  (setv r 0)\n  (for [t (range (len df))]\n    (+= r (-> (conf-model a alpha t0 t) (- (get df t)) (** 2))))\n  r)\n\n(-> conf-model-loss \n    (minimize :x0 (np.array [0.1 1.0 5]) \n              :args conf-actual \n              :method "Nelder-Mead" :tol 1e-6)\n    (s1 conf-opt))')




get_ipython().run_cell_magic('h', '', '\n; Define dead model then run it.\n\n(defn dead-model [death-rate lag t]\n  (s (, a alpha t0) conf-opt.x)\n  (s t (- t lag))\n  (s conf (conf-model a alpha t0 t))\n  (s dead (* conf death-rate)))\n\n(defn dead-model-loss [x df]\n  (s (, death-rate lag) x)\n  (s (, a alpha t0) conf-opt.x)\n  (s r 0)\n  (for [t (range (len df))]\n    (+= r (-> (dead-model death-rate lag t) (- (get df t)) (** 2))))\n  r)\n\n(-> dead-model-loss \n    (minimize :x0 (np.array [0.01 15]) \n              :args dead-actual\n              :method "Nelder-Mead" :tol 1e-6)\n    (s1 dead-opt))\n\n[conf-opt dead-opt]')




get_ipython().run_cell_magic('h', '', '(defn model-to-fn [model opt] \n  (fn [&rest args]\n    (setv params (-> opt (. x) (list) (+ (list args))))\n    (-> model (apply params))))\n\n(-> conf-model (model-to-fn conf-opt) (s1 conf-fn))\n(-> dead-model (model-to-fn dead-opt) (s1 dead-fn))')




get_ipython().run_cell_magic('h', '', '; Compare actual vs predictions\n(-> conf-actual (len) (range) (map1 conf-fn) (pd.Series :name "Predict") (s1 conf-predict))\n(-> (pd.concat [conf-actual conf-predict] :axis 1) (s1 conf-eval))\n(-> conf-eval (* population) (.plot :title "Confirmed Cases"))\n\n(-> dead-actual (len) (range) (map1 dead-fn) (pd.Series :name "Predict") (s1 dead-predict))\n(-> (pd.concat [dead-actual dead-predict] :axis 1) (s1 dead-eval))\n(-> dead-eval (* population) (.plot :title "Fatalities"))')




get_ipython().run_cell_magic('h', '', '(import [sklearn [metrics]])\n\n; Calculate conf errors.\n(print "Confirmed Cases Errors")\n(print "Conf MSE =" (metrics.mean-squared-error (conf-eval.Actual.to-numpy) (conf-eval.Predict.to-numpy)))\n(print "Conf MAE =" (metrics.mean-absolute-error (conf-eval.Actual.to-numpy) (conf-eval.Predict.to-numpy)))\n(print "Conf RMSE =" (np.sqrt (metrics.mean-squared-error (conf-eval.Actual.to-numpy) (conf-eval.Predict.to-numpy))))\n\n; Calculate dead errors.\n(print "Fatalities Errors")\n(print "Dead MSE =" (metrics.mean-squared-error (dead-eval.Actual.to-numpy) (dead-eval.Predict.to-numpy)))\n(print "Dead MAE =" (metrics.mean-absolute-error (dead-eval.Actual.to-numpy) (dead-eval.Predict.to-numpy)))\n(print "Dead RMSE =" (np.sqrt (metrics.mean-squared-error (dead-eval.Actual.to-numpy) (dead-eval.Predict.to-numpy))))')




get_ipython().run_cell_magic('h', '', '; Next lets build out the test\n(defn pd-head-tail [df]\n  (print "Rows = "(-> df (len)))\n  (-> df (.head 1) (display))\n  (-> df (.tail 1) (display)))\n\n(-> covid-train  (pd.read-csv) (s1 df-train))\n(-> covid-test   (pd.read-csv) (s1 df-test))\n\n;(pd-head-tail df-train)\n;(pd-head-tail df-test)')




get_ipython().run_cell_magic('h', '', '; Compute date0 of training data.\n(-> df-train (. Date) (get 0) (dateparser.parse) (s1 date0-train))\n\n; Useful functions.\n(defn date-string->t [d] \n  (-> d (dateparser.parse) (- date0-train) (. days)))\n(defn date-string->confirmed-cases [d]\n  (-> d (date-string->t) (conf-fn) (* population) (int)))\n(defn date-string->fatalities [d]\n  (-> d (date-string->t) (dead-fn) (* population) (int)))\n\n; Submission.\n(-> df-test \n    (.to-dict "records") \n    (map1 (fn [r] \n            (-> r\n              (assoc1 "ConfirmedCases" (-> r (get "Date") (date-string->confirmed-cases)))\n              (assoc1 "Fatalities"     (-> r (get "Date") (date-string->fatalities))))))\n    (pd.DataFrame)\n    (pd-keep ["ForecastId" "ConfirmedCases" "Fatalities"])\n    (.to-csv "submission.csv" :index False))')






