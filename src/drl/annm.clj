(ns drl.annm
  (:require [clojure.core.matrix :refer :all]
            [clojure.java.io :as io]
            [drl.utils :refer [roundn zipv dotp sigmoid zerov]]))

;;;-----------------------------------------------------------------------

;;; Use fast matrix operations.

(set-current-implementation :vectorz)

;;;-----------------------------------------------------------------------
;;; ANN Parameters
;;;-----------------------------------------------------------------------

(def learning-rate "Learning rate" 0.1)
(def max-weight "Max weight" 0.1)


;;;-----------------------------------------------------------------------
;;; Display functions
;;;-----------------------------------------------------------------------

(defn show-vector [v] (println (mapv roundn v)))

(defn show-matrix [weights](mapv (fn [row] (show-vector row)) weights))

;;;-----------------------------------------------------------------------
;;; Bias vector and Weight Matrix
;;;-----------------------------------------------------------------------

(defn random-weight [] (roundn (- (rand (* 2 max-weight)) max-weight)))

;;;-----------------------------------------------------------------------

(defn weightv "Creates and returns a new weight vector" [n]
  (array (mapv (fn [x](random-weight)) (range n))))

;;;-----------------------------------------------------------------------

(defn weightm "Creates and returns a new weight matrix" [n m]
  (array (mapv (fn [x] (mapv (fn [y](random-weight))(range n))) (range m))))

;;;-----------------------------------------------------------------------

(defn v-net-to-m-net [v-net]
  (map (fn [layer] (list (array (first layer))(array (second layer))))
       v-net))

;;;-----------------------------------------------------------------------
;;; Network Generation and Computation
;;;-----------------------------------------------------------------------

;;; Returns a bias vector and a weights matrix
(defn gen-layer [[n-in m-out]] [(weightv m-out) (weightm n-in m-out)])

;;; Returns a vector of bias vetors and weight matrices
;;; <net-specs> is of the for [[n-in n-out]*]

(defn gen-net [net-specs](mapv gen-layer net-specs))

;;;-----------------------------------------------------------------------
;;; Layer Outputs
;;;-----------------------------------------------------------------------

;;; Output vector O of layer k: Ok = σ (WkOk-1 + Bk)

(defn layer-outputs [inputs [bias weights]]
  (emap sigmoid (add bias (mmul weights inputs))))

;;;-----------------------------------------------------------------------
;;; Compute Network
;;;-----------------------------------------------------------------------

;;; Returns a vector of the outputs of each layer

(defn compute-network [inputs net]
  (reduce (fn [acc layer] (conj acc (layer-outputs (last acc) layer)))
          [inputs]
          net))

;;;-----------------------------------------------------------------------
;;; Network Ouputs
;;;-----------------------------------------------------------------------

;;; Returns the output of the network for the speecified <input>.

(defn network-outputs [inputs net]
  (last (compute-network inputs net)))

;;;-----------------------------------------------------------------------
;;; Network Errors
;;;-----------------------------------------------------------------------

;;; δk = ok (1 – ok) (tk – ok)

;;; <outputs> are the network outputs and targets are the desired values.

(defn output-errors [outputs targets]
  (mul outputs (sub 1 outputs)(sub targets outputs)))

;;;-----------------------------------------------------------------------

;;; δh = oh (1 – oh)  Σ (whk δk )

(defn layer-errors [outputs weights errors]
  (mul outputs (sub 1 outputs)(mmul (transpose weights) errors)))

;;;-----------------------------------------------------------------------

;;; We Reverse network and outputs to compute errors of each layer.

(defn network-errors [net [training-inputs training-outputs]]
  (let [net-outputs (compute-network training-inputs net)
        r-net (reverse net)
        r-out (rest (reverse (rest net-outputs)))
        zip-net (zipv (rest r-net) r-net)
        net-errors
        (reverse
         (reduce (fn [acc [[[n-bias n-weights][p-bias p-weights]] n-out]]
                   (conj acc (layer-errors n-out p-weights (last acc))))
                 [(output-errors (last net-outputs) training-outputs)]
                 (zipv zip-net r-out)))]
    net-errors))

;;;-----------------------------------------------------------------------
;;; Repeating Inputs and Errors
;;;-----------------------------------------------------------------------

(defn inputs-matrix [inputs size]
  (array (take size (repeat (vec inputs)))))

;;;-----------------------------------------------------------------------

(defn errors-matrix [errors size]
  (array (take size (repeat (vec errors)))))

;;;-----------------------------------------------------------------------
;;; Update F`unctions
;;;-----------------------------------------------------------------------

(defn update-bias-vector [biasv errors]
 (add biasv (mul learning-rate errors)))
 
;;;-----------------------------------------------------------------------

;;; wij = wij + η δh xij 

(defn update-weights [inputs weights-matrix errors]
  (let [errorm (transpose (errors-matrix errors (second (shape inputs))))]
    (add weights-matrix (mul learning-rate (mul inputs errorm)))))

;;;-----------------------------------------------------------------------
;;; Update Layer
;;;-----------------------------------------------------------------------

;;; Returns a new layer witth updated bias and weights.

(defn update-layer [inputs bias-vector weights-matrix errors]
  [(update-bias-vector bias-vector errors)
   (update-weights (inputs-matrix inputs (first (shape weights-matrix)))
                   weights-matrix
                   errors)])

;;;-----------------------------------------------------------------------
;;; Update Network
;;;-----------------------------------------------------------------------

(defn update-network [net [training-inputs training-outputs]]
  (let [net-errors (network-errors net [training-inputs training-outputs])]
    (first
     (reduce (fn [[acc inputs] [[bias weights] errors]]
               [(conj acc (update-layer inputs bias weights errors))
                (layer-outputs inputs [bias weights])])
             [[] training-inputs]
             (zipv net net-errors)))))

;;;-----------------------------------------------------------------------
;;; BACKPROP
;;;-----------------------------------------------------------------------

;;; This simply iterates over the training data

(defn backprop-1 [current-net td]
  (reduce (fn [net ti] (update-network net ti)) current-net td))

;;;-----------------------------------------------------------------------

;;; This iterates the specified number of times

(defn backprop [initial-net td iterations]
  (reduce (fn [net iteration] (backprop-1 net td)) initial-net (range iterations)))

;;;-----------------------------------------------------------------------
;;; Network Display
;;;-----------------------------------------------------------------------

(defn show-layer [layer-number [bias weights]]
  (let [bias (mapv roundn bias)]
    (println (str "\nLayer " layer-number ":"))
    (println "Bias vector:" bias)
    (println "\nWeights:")
    (show-matrix weights)
    nil))

;;;-----------------------------------------------------------------------


(defn show-net [net] (mapv show-layer (range 2 (+ 2 (count net))) net))

;;;-----------------------------------------------------------------------
;;; VERIFY-TRAINING-ACCURACY
;;;-----------------------------------------------------------------------

(defn get-training-accuracy [net td]
  (reduce (fn [[correct incorrect][ti-inputs ti-outputs]]
            (let [net-outputs (network-outputs ti-inputs net)
                  rounded-outputs (mapv #(int (roundn % 0)) net-outputs)]
              (if (= rounded-outputs ti-outputs)
                [(inc correct) incorrect]
                [correct (inc incorrect)])))
          [0 0]
          td))

;;;-----------------------------------------------------------------------

(defn verify-training-accuracy [net td]
  (let [[correct incorrect] (get-training-accuracy net td)]
    (println "Total training:    " (count td))
    (println "Total correct:     " correct)
    (println "Total incorrect:   " incorrect)
    (println "Percentage correct:" (* 100.0 (/ correct (count td))))
    [correct incorrect]))

;;;-----------------------------------------------------------------------
;;; VERIFY-TRAINING-EXAMPLES
;;;-----------------------------------------------------------------------

(defn verify-training-examples [net td]
  (mapv (fn [[ti-inputs ti-outputs]]
          (let [net-outputs (network-outputs ti-inputs net)
                rounded-outputs (mapv #(int (roundn % 0)) net-outputs)]
            (println "Actual Training Inputs:  " ti-inputs)
            (println "Actual Training Outputs: " ti-outputs)
            (println "Rounded Network Outputs: " rounded-outputs)
            (println "Actual Network Outputs:  " net-outputs)
            (println "\n")))
        td)
  nil)

;;;-----------------------------------------------------------------------
;;; End of file
;;;-----------------------------------------------------------------------

