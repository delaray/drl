(ns drl.charts
  (:require [clojure.string :as str]
            [incanter.charts :as c]
            [incanter.core :as i])

;;;-----------------------------------------------------------------------
;;; Test-Points
;;;-----------------------------------------------------------------------

(def ANN-1-300
  [[0 	 0.40]
   [50	19.26]
   [100 19.20]
   [150 20.03]
   [200 17.11]
   [250 17.98]
   [300 18.09]
   [350 18.15]
   [400 18.16]
   [450 18.23]
   [500 18.45]
   [550 18.5]])

;;;-----------------------------------------------------------------------

(defn test-points
  [points]
  (let [iterations (map first points)
        percent-correct (map second points)
        plot (c/line-chart iterations percent-correct)]
     (i/view plot)))

;;;-----------------------------------------------------------------------
;;; Plot-MLP
;;;-----------------------------------------------------------------------

(defn plot-mlp [mlp1 mlp2 mlp3]
   (let [iterations (mapv first mlp1)
         correct (mapv second mlp1)
         plot (c/xy-plot nil nil :x-label "Iterations" :y-label "Total Correct")]
     (c/add-lines plot (mapv first mlp1)(mapv second mlp1))
     (c/add-lines plot (mapv first mlp2)(mapv second mlp2))
     (c/add-lines plot (mapv first mlp3)(mapv second mlp3))
     (i/view plot)
     plot))
 
;;;-----------------------------------------------------------------------
;;; End of File
;;;-----------------------------------------------------------------------



