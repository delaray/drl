(ns drl.annv
  (:require [clojure.java.io :as io]
            [drl.utils :refer [roundn zipv dotp transpose sigmoid zerov]]))

;;;-----------------------------------------------------------------------
;;; ANN Parameters
;;;-----------------------------------------------------------------------

(def lr 0.1)
(def max-weight 0.1)

;;;-----------------------------------------------------------------------
;;; Weights
;;;-----------------------------------------------------------------------

(defn random-weight [] (roundn (- (rand (* 2 max-weight)) max-weight)))

(defn weightv "Returns weight vector" [n](mapv (fn [x](random-weight)) (range n)))

(defn weightm "Returns weight matrix" [n m](mapv (fn [x](weightv n)) (range m)))

;;;-----------------------------------------------------------------------
;;; Network Generation and Computation
;;;-----------------------------------------------------------------------

;;; Returns a bias vector and weights matrix
(defn gen-layer [[inputs outputs]] [(weightv outputs) (weightm inputs outputs)])

;;; Returns a network of bias vetors and weight matrices
(defn gen-net [net-specs](mapv gen-layer net-specs))

;;;-----------------------------------------------------------------------
;;; Network Computation
;;;-----------------------------------------------------------------------

(defn layer-outputs [inputs [bias weights]]
   (mapv (fn [b wv](roundn (sigmoid (+ b (dotp inputs wv))))) bias weights))

;;;-----------------------------------------------------------------------

(defn network-outputs [inputs net]
  (reduce (fn [inputs next-layer] (layer-outputs inputs next-layer)) inputs net))

;;;-----------------------------------------------------------------------

;;; Returns a vector of the outputs of each layer

(defn compute-network [inputs net]
  (reduce (fn [acc layer] (conj acc (layer-outputs (last acc) layer)))
          [inputs]
          net))

;;;-----------------------------------------------------------------------
;;; Network Errors
;;;-----------------------------------------------------------------------

(defn output-errors
  "Returns a vector of network output errors"
  [outputs targets]
  (mapv (fn [o t] (roundn (* o (- 1 o)(- t o)))) outputs targets))

;;;-----------------------------------------------------------------------

(defn layer-errors [outputs weights errors]
  (mapv (fn [w o] (roundn (* o (- 1 o)(dotp w errors))))
        (transpose weights) outputs))

;;;-----------------------------------------------------------------------

(defn show-vector [v] (println (mapv roundn v)))

(defn show-matrix [matrix]
(mapv (fn [row] (show-vector row)) matrix))

;;;-----------------------------------------------------------------------

;;; Reverse network to compute errors of each layer.

(defn network-errors [net [training-inputs training-outputs]]
  (let [net-outputs (network-outputs training-inputs net)
        net-errors (output-errors net-outputs training-outputs)
        rnet (reverse net)
        net-errors
        (reverse
         (reduce (fn [acc [[n-bias n-weights][p-bias p-weights]]]
                   (let [outputs (layer-outputs training-inputs [n-bias n-weights])]
                     (conj acc (layer-errors outputs p-weights (last acc)))))
                 [net-errors]
                 (zipv (rest rnet) rnet)))]
    ;;(println "Network errors:")
    ;;(mapv println net-errors)
    net-errors))

;;;-----------------------------------------------------------------------

(defn show-update-weights-args  [inputs weights-matrix errors]
  (println "-------------------------------------------\n")
  (println "Update weights args:\n")
  (println "Inputs: ")
  (show-vector inputs)
  (println "Weights: ")
  (show-matrix weights-matrix)
  (println "Errors: ")
  (show-vector errors))

;;;-----------------------------------------------------------------------
;;; Bias vector and weights  Update
;;;-----------------------------------------------------------------------

(defn update-bias-vector [biasv errors]
  (mapv (fn [bias error](roundn (+ bias (* lr error)))) biasv errors))

;;;-----------------------------------------------------------------------



(defn update-rule [inputs weights error]
  (let [delta (mapv (fn [i] (roundn (* lr i error))) inputs)]
    ;;(println "Delta: ")
    ;;(println delta)
    (mapv (fn [w d] (roundn (+ w d))) weights delta)))
  

;;;-----------------------------------------------------------------------
;;; UPDATE-WEIGHTS
;;;-----------------------------------------------------------------------

(defn update-weights [inputs weights errors]
  ;;(show-update-weights-args inputs weights errors)
  (mapv (fn [weightv error] (update-rule inputs weightv error)) weights errors))

;;;-----------------------------------------------------------------------
;;; Update Layer
;;;-----------------------------------------------------------------------

(defn update-layer [inputs biasv weights errors]
  ;;(println "----------------------------------------------------")
  ;;(println "Layer")
  ;; (println "----------------------------------------------------")
  [(update-bias-vector biasv errors)
   (update-weights inputs weights errors)])

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

;;; This trains the networwork once with one training instance.

(defn backprop-2 [net training-instance]
  (update-network net training-instance))

;;;-----------------------------------------------------------------------

;;; This simply iterates over the training data

(defn backprop-1 [current-net td]
  (reduce (fn [net ti] (backprop-2 net ti)) current-net td))

;;;-----------------------------------------------------------------------

;;; This iterates the specified number of times

(defn backprop [initial-net td iterations]
  (reduce (fn [[net count] iteration]
            [(backprop-1 net td)(inc count)])
          [initial-net 0]
          (range iterations)))

;;;-----------------------------------------------------------------------
;;; Network Display
;;;-----------------------------------------------------------------------

(defn show-layer [layer-number [bias weights]]
  (println (str "\nLayer " layer-number ":"))
  (println "Bias vector:" bias)
  (println "\nWeights:")
  (mapv println weights)
  nil)

;;;-----------------------------------------------------------------------

(defn show-net [net] (mapv show-layer (range 2 (+ 2 (count net))) net))

;;;-----------------------------------------------------------------------
;;; VERIFY-TRAINING-ACCURACY
;;;-----------------------------------------------------------------------

(defn verify-training-accuracy [net td]
  (let [[correct incorrect]
        (reduce (fn [[correct incorrect][ti-inputs ti-outputs]]
                  (let [net-outputs (network-outputs ti-inputs net)
                        rounded-outputs (mapv #(int (roundn % 0)) net-outputs)]
                    (if (= rounded-outputs ti-outputs)
                      [(inc correct) incorrect]
                      [correct (inc incorrect)])))
                [0 0]
                td)]
    (println "Total training:    " (count td))
    (println "Total correct:     " correct)
    (println "Total incorrect:   " incorrect)
    (println "Percentage correct:" (* 100.0 (/ correct (count td))))))

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
