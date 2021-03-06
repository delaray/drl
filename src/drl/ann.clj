(ns drl.ann
  (:require [drl.utils :refer [roundn dotp zipv transpose sigmoid zerov]]))

;;;-----------------------------------------------------------------------
;;; ANN Parameters
;;;-----------------------------------------------------------------------

(def lr 0.3)
(def max-weight 0.3)

;;;-----------------------------------------------------------------------
;;; Utilities
;;;-----------------------------------------------------------------------

(defn random-weight [] (roundn (- (rand (* 2 max-weight)) max-weight)))

;;;-----------------------------------------------------------------------

(defn weightv "Returns  n weights" [n](mapv (fn [x](random-weight)) (range n)))

;;;***********************************************************************
;;; Part 1: ANN Creation
;;;***********************************************************************

(defn make-layer-name [layer-number]
  (keyword (str "l" layer-number)))

;;;-----------------------------------------------------------------------
;;; GEN-UNIT, GEN-LAYER & GEN-NET
;;;-----------------------------------------------------------------------

(defn gen-unit [unit-number number-of-inputs]
  {:unit unit-number
   :inputs (zerov number-of-inputs)
   :weights (weightv number-of-inputs)
   :output 0})

;;;-----------------------------------------------------------------------

(defn gen-layer [number-of-units number-of-inputs]
  (mapv (fn [unit-number] (gen-unit unit-number number-of-inputs))
        (range number-of-units)))

;;;-----------------------------------------------------------------------

;; <layers-specs> is of the form: [<layer> <units>]

(defn gen-net [layers]
  (reduce (fn [net [layer units inputs]]
            (merge net {(make-layer-name layer)(gen-layer units inputs)}))
          (sorted-map)
          layers))

;;;-----------------------------------------------------------------------
;;; Units
;;;-----------------------------------------------------------------------

(defn unit-name [unit]
  (first (keys unit)))

(defn unit-value [unit]
  (first (vals unit)))

(defn unit-inputs [unit-val]
  (:inputs unit-val))

(defn unit-weights [unit-val]
  (:weights unit-val))

;;;-----------------------------------------------------------------------
;;; Layers
;;;-----------------------------------------------------------------------

(defn layer-name [layer]
  (first (keys layer)))

(defn layer-units [layer]
  (first (vals layer)))

;;;-----------------------------------------------------------------------
;;; Inputs
;;;-----------------------------------------------------------------------

(defn input-units [net]
  (net (make-layer-name 1)))

;;;-----------------------------------------------------------------------
;;; Hidden Layers
;;;-----------------------------------------------------------------------

(defn hidden-layers [net]
  (mapv (fn [layer-number]
          (let [layer-name (make-layer-name layer-number)]
            {layer-name (net layer-name)}))
        ;; Hidden unit layer numbers
        (butlast (range 2 (inc (inc (count net)))))))

;;;-----------------------------------------------------------------------

;;; These functions assues a single hidden unit layer.

(defn hidden-layer-name [net]
  (first (keys (first (hidden-layers net)))))

(defn hidden-units [net]
  (first (vals (first (hidden-layers net)))))

;;;-----------------------------------------------------------------------
;;; Output Layer
;;;-----------------------------------------------------------------------

(defn output-layer-name [net]
  (make-layer-name (inc (count net))))

(defn output-units [net]
  (net (output-layer-name net)))

(defn network-outputs [net]
  (mapv :output (vals (output-units net))))

;;;***********************************************************************
;;; Part 2: Computing and Network Values
;;;***********************************************************************

(defn compute-unit-output [unit]
  (roundn (sigmoid (dotp (:inputs unit)(:weights unit)))))

(defn update-unit-inputs [unit inputs]
  (let [unit (assoc unit :inputs inputs)]
    (assoc unit  :output (compute-unit-output unit))))

(defn update-unit-weights [unit weights]
  (let [unit (assoc unit :weights weights)]
    (assoc unit  :output (compute-unit-output unit))))

(defn layer-outputs [units] (mapv :output units))

(defn network-outputs [net]
  (layer-outputs (net (output-layer-name net))))

;;;-----------------------------------------------------------------------

(defn update-layer-inputs [units inputs]
  (mapv (fn [unit] (update-unit-inputs unit inputs)) units))

;;;-----------------------------------------------------------------------

;;; Apllies inputs to entire network

(defn compute-network [net inputs]
  (first (reduce (fn [[net inputs] key]
                   (let [new-units (update-layer-inputs (net key) inputs)
                         new-net (assoc net key new-units)]
                     [new-net (layer-outputs new-units)]))
                 [net inputs]
                 (keys net))))

;;;***********************************************************************
;;; Backprogation
;;;***********************************************************************

;;;-----------------------------------------------------------------------
;;; Unit Errors
;;;-----------------------------------------------------------------------

;;; Returns an error vector of output unit errors.

(defn output-unit-errors [net target-outputs]
  (mapv (fn [net-o ti-o] (roundn (* net-o (- 1 net-o)(- ti-o net-o))))
        (network-outputs net)
        target-outputs))

;;;-----------------------------------------------------------------------
;;; Hidden Unit Errors
;;;-----------------------------------------------------------------------

;;; Returns the error of the hidden-unit.

(defn hidden-unit-error [net hidden-unit weight-vector errors]
  (let [unit-output (:output hidden-unit)]
    (roundn (* unit-output (- 1 unit-output)(dotp weight-vector errors)))))

;;;-----------------------------------------------------------------------

;;; Returns a vector of hidden unit errors.

(defn hidden-unit-errors [net ti-outputs]
  (let [weight-vectors (transpose (mapv :weights (output-units net)))
        errors (output-unit-errors net ti-outputs)]
    (reduce (fn [acc [unit weight-vector]]
              (conj acc (hidden-unit-error net unit weight-vector errors)))
            []
            (zipv (hidden-units net) weight-vectors))))

;;;-----------------------------------------------------------------------
;;; Unit Updates
;;;-----------------------------------------------------------------------

(defn update-rule [inputs weights error]
  (mapv (fn [i w] (roundn (+ w (* lr i error)))) inputs weights))

;;;-----------------------------------------------------------------------

(defn update-output-units [net ti-outputs]
  (assoc net (output-layer-name net)
         (reduce (fn [acc [unit  unit-error]]
                   (let [inputs (unit-inputs unit)
                         weights (unit-weights unit)
                         new-weights (update-rule inputs weights unit-error)]
                     (conj acc (update-unit-weights unit new-weights))))
                 []
                 (zipv (output-units net) (output-unit-errors net ti-outputs)))))

;;;-----------------------------------------------------------------------

;;; [inputs outputs] is the destructured training instance

(defn update-hidden-units [net ti-outputs]
  ;;(println "\nHidden Unit Errors: " hidden-errors)
  (assoc net (hidden-layer-name net)
         (reduce (fn [acc [unit unit-error]]
                   (let [inputs (unit-inputs unit)
                         weights (unit-weights unit)
                         new-weights (update-rule inputs weights unit-error)]
                     (conj acc (update-unit-weights unit new-weights))))
                 []
                 (zipv (hidden-units net)(hidden-unit-errors net ti-outputs)))))

;;;-----------------------------------------------------------------------
;;; BACKPROP
;;;-----------------------------------------------------------------------

;;; This trains the networwork once with one training instance.

(defn backprop-2 [current-net training-instance]
  (let [[ti-inputs ti-outputs] training-instance
        computed-net (compute-network current-net ti-inputs)]
    (let [updated-net-1 (update-output-units computed-net ti-outputs)
          updated-net-2 (update-hidden-units updated-net-1 ti-outputs)]
      updated-net-2)))

;;;-----------------------------------------------------------------------

;;; This simply iterates over the training data

(defn backprop-1 [current-net td]
  (reduce (fn [net ti] (backprop-2 net ti)) current-net td))

;;;-----------------------------------------------------------------------

;;; This iterates the specified number of times

(defn backprop [initial-net td iterations]
  (reduce (fn [net iteration] (backprop-1 net td)) initial-net (range iterations)))

;;;***********************************************************************
;;; Display Functions
;;;***********************************************************************

(defn layer-units [layer] (first (vals layer)))

(defn show-layer [layer-name units]
  (println (str "\nLayer " (name layer-name) ":"))
  (mapv (fn [unit]
          (println (str "Unit" (:unit unit) ":") (dissoc unit :unit)))
        units))

;;;-----------------------------------------------------------------------

(defn show-net [net]
  (mapv #(show-layer % (net %)) (keys net))
  (println "\n")
  true)

;;;-----------------------------------------------------------------------
;;; VERIFY-TRAINING-EXAMPLES
;;;-----------------------------------------------------------------------

(defn verify-training-examples [net td]
  (mapv (fn [[ti-inputs ti-outputs]]
          (let [net-outputs (network-outputs (compute-network net ti-inputs))
                rounded-outputs (mapv #(int (roundn % 0)) net-outputs)]
            (println "Actual Training Inputs:  " ti-inputs)
            (println "Actual Training Outputs: " ti-outputs)
            (println "Rounded Network Outputs: " rounded-outputs)
            (println "Actual Network Outputs:  " net-outputs)
            (println "\n")))
        td)
  nil)

;;;***********************************************************************
;;; IDENTITY FUNCTION EXAMPLE
;;;***********************************************************************

(def if-net (gen-net [[2 3 8][3 8 3]]))

;;; Training Data Identity function example.

(def if-td
    [[[1 0 0 0 0 0 0 0] [1 0 0 0 0 0 0 0]]
     [[0 1 0 0 0 0 0 0] [0 1 0 0 0 0 0 0]]
     [[0 0 1 0 0 0 0 0] [0 0 1 0 0 0 0 0]]
     [[0 0 0 1 0 0 0 0] [0 0 0 1 0 0 0 0]]
     [[0 0 0 0 1 0 0 0] [0 0 0 0 1 0 0 0]]
     [[0 0 0 0 0 1 0 0] [0 0 0 0 0 1 0 0]]
     [[0 0 0 0 0 0 1 0] [0 0 0 0 0 0 1 0]]
     [[0 0 0 0 0 0 0 1] [0 0 0 0 0 0 0 1]]])


;;;-----------------------------------------------------------------------
;;; End of File
;;;-----------------------------------------------------------------------
