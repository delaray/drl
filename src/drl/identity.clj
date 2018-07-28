(ns drl.identity
  (:require  [clojure.core.matrix :refer :all]
             [drl.annv :as v]
             [drl.annm :as m]))

;; [backprop gen-net show-net verify-training-accuracy verify-training-examples]]))

;;;-----------------------------------------------------------------------
;;; IDENTITY FUNCTION EXAMPLE
;;;-----------------------------------------------------------------------

;; 1 Hidden Layer: Require 4,500 iterations
(def if-net-v-1 (v/gen-net [[8 3][3 8]]))
(def if-net-m-1 (m/v-net-to-m-net if-net-v-1))

;; 2 Hidden Layers: Requires 28,000 iterations to converge
(def if-net-v-2 (v/gen-net [[8 3][3 3][3 8]]))
(def if-net-m-2 (m/v-net-to-m-net if-net-v-2))

;; 3 Hidden Layers: Requires 200,000+ iterations to converge
(def if-net-v-3 (v/gen-net [[8 3][3 3][3 3][3 8]]))
(def if-net-m-3 (m/v-net-to-m-net if-net-v-3))

;;;-----------------------------------------------------------------------

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
;;; Clojure Vectors Based Example
;;;-----------------------------------------------------------------------

(def default-iterations 100000)

;;; Training function

(defn v-train-identity
  ([] (v-train-identity default-iterations))
  ;; One iteration on a single training instance
  ([iterations] (first (v/backprop if-net-v-1 if-td iterations))))

;;;-----------------------------------------------------------------------

(defn run-vector-example 
  ([]
   (run-vector-example default-iterations))
  ([iterations]
   (let [tnv (v-train-identity iterations)]
     ;; Print training examples output
     (v/verify-training-examples tnv if-td)
     ;; Print accuracy results
     (v/verify-training-accuracy tnv if-td)
     ;; Return the trained network
     tnv)))

;;;-----------------------------------------------------------------------
;;; Fast Matrix Based Example
;;;-----------------------------------------------------------------------

;;; Training function

(defn m-train-identity
  ([] (m-train-identity default-iterations))
  ;; One iteration on a single training instance
  ([iterations] (m-train-identity if-net-m-1 iterations))
  ([net iterations](m/backprop net if-td iterations)))

;;;-----------------------------------------------------------------------

(defn run-matrix-example
  ([](run-matrix-example default-iterations))
  ([iterations]
   (let [tnm (m-train-identity iterations)]
    ;; Print training examples output
    (m/verify-training-examples tnm if-td)
    ;; Print accuracy results
    (m/verify-training-accuracy tnm if-td)
    ;; Return the trained network
    tnm)))
   
;;;-----------------------------------------------------------------------

(defn compute-convergence-m [net]
  (let [iterations 500
        max-iterations 1000000
        x-points (rest (for [i (range max-iterations) :when (= (mod i iterations) 0)] i))
        data-points 
        (reduce (fn [acc nv]
                  (let [tn (nth (last acc) 2)
                        new-tn (m-train-identity tn iterations)
                        correct (first (m/get-training-accuracy new-tn if-td))]
                    (conj acc [nv correct new-tn])))
                [[0 0 net]]
                x-points)]
    (mapv (fn [[x y _]][x y]) data-points)))

;;;-----------------------------------------------------------------------
;;; End of file
;;;-----------------------------------------------------------------------
