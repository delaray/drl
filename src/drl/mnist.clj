(ns drl.mnist
  (:require  [clojure.data.csv :as csv]
             [clojure.java.io :as io]
             [clojure.core.matrix :refer :all]
             [drl.annm :refer [backprop gen-net show-net verify-training-accuracy]]))

;;;---------------------------------------------------------------------------
;;; Loading MNIST Traiing and Test Data
;;;---------------------------------------------------------------------------

(defn load-csv
  "Loads specified csv file as a vector of vectors."
  [filename]
  (with-open [in-file (io/reader filename)]
    (doall (csv/read-csv in-file :separator \,))))

;;;---------------------------------------------------------------------------

(def mnist-training-file "data/mnist_train.csv")
(def mnist-test-file "data/mnist_test.csv")

;;; Note CSV Format is digit followed by 784 pixel values (28x28)

(defn load-mnist-training-data [](mapv #(mapv read-string %)(load-csv mnist-training-file)))
(defn load-mnist-test-data [] (mapv #(mapv read-string %)(load-csv mnist-test-file)))

(def target-outputs-map
  {0 [1 0 0 0 0 0 0 0 0 0],
   1 [0 1 0 0 0 0 0 0 0 0],
   2 [0 0 1 0 0 0 0 0 0 0],
   3 [0 0 0 1 0 0 0 0 0 0],
   4 [0 0 0 0 1 0 0 0 0 0],
   5 [0 0 0 0 0 1 0 0 0 0],
   6 [0 0 0 0 0 0 1 0 0 0],
   7 [0 0 0 0 0 0 0 1 0 0],
   8 [0 0 0 0 0 0 0 0 1 0],
   9 [0 0 0 0 0 0 0 0 0 1]})

(defn gen-mnist-training-data
  ([] (gen-mnist-training-data 60000))
  ([n](take n (mapv (fn [row][(vec (rest row))(target-outputs-map (first row))])
                    (load-mnist-training-data)))))

(defn gen-mnist-test-data [n]
  (take n (mapv (fn [row][(vec (rest row))(target-outputs-map (first row))])
                (load-mnist-test-data))))

(def mnist-training-data (gen-mnist-training-data))

;;;---------------------------------------------------------------------------
;;; MNIST ANN
;;;---------------------------------------------------------------------------

;; (def mnist-test-data (gen-mnist-test-data))

;;; 1 Hidden Layer with 100 units
(def mnist-net (gen-net [[784 100][100 10]]))
(def mnist-net-1-100 (gen-net [[784 100][100 10]]))

;;; 1 Hidden Layer with 200 units
(def mnist-net-1-200 (gen-net [[784 200][200 10]]))

;;; 2 Hidden Layers with 100 units
(def mnist-net-2-100 (gen-net [[784 100][100 100][100 10]]))

;;; 3 Hidden Layers with 100 units
(def mnist-net-3-100 (gen-net [[784 100][100 100][100 100][100 10]]))

;;; 1 Hidden Layer with 300 units
(def mnist-net-1-300 (gen-net [[784 300][300 10]]))

(defn train-mnist [net iterations]
  (backprop net mnist-training-data iterations))


(defn ms-to-mn [ms] (/ ms 1000.0 60.0))

;;;---------------------------------------------------------------------------

;;; Runs <trials> number training sessions each consisting of
;;; <iterations> number of iterations.

(defn run-mnist [net iterations trials]
  (reduce (fn [net trial]
            (let [tn (time (train-mnist net iterations))]
              (println "-----------------------------------------")
              (println "Trial #" trial ", Iterations: " iterations)
              (verify-training-accuracy tn mnist-training-data)
              tn))
          net
          (range trials)))

;;;---------------------------------------------------------------------------
;;; End of File
;;;---------------------------------------------------------------------------
