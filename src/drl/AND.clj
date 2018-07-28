(ns drl.AND
  (:require [clojure.core.matrix :refer :all]
            [drl.annm :refer :all]))

;;;-----------------------------------------------------------------------

;;; Use fast matrix operations.

(set-current-implementation :vectorz)

;;;-----------------------------------------------------------------------
;;; AND FUNCTION EXAMPLE
;;;-----------------------------------------------------------------------

(def and-net (gen-net [[2 3][3 1]]))

;;; Training Data for AND function.

(def and-td
  [[(array [1 0]) (array [0])]
   [(array [1 1]) (array [1])]
   [(array [0 1]) (array [0])]
   [(array [0 0]) (array [0])]])

(def ti (first (first and-td)))
(def to (second (first and-td)))

(def t [ti to])

(def no (network-outputs ti and-net))

(def errs (network-errors and-net t))

;;; Training function

(defn train-and [] (backprop and-net and-td 10000))


;;; (def tn (first (train-and)))
;;; (verify-training-accury tn td)
;;; (verify-training-examples tn td)

;;;-----------------------------------------------------------------------
;;; End of file
;;;-----------------------------------------------------------------------
