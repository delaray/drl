(ns drl.cnn
  (:require [clojure.data.csv :as csv]
            [clojure.java.io :as io]
            [clojure.core.matrix :refer :all]
            [drl.utils :refer [roundn zipv dotp sigmoid zerov]]))


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

;;;---------------------------------------------------------------------------

(defn gen-mnist-training-data [n]
  (take n (mapv (fn [row][(vec (rest row))(target-outputs-map (first row))])
                (load-mnist-training-data))))

(defn gen-mnist-test-data [n]
  (take n (mapv (fn [row][(vec (rest row))(target-outputs-map (first row))])
                (load-mnist-test-data))))

(def mnist-training-data (gen-mnist-training-data 6000))

;;;-----------------------------------------------------------------------
;;; Bias vector and Weight Matrix
;;;-----------------------------------------------------------------------

(defn random-weight [] (roundn (- (rand (* 2 max-weight)) max-weight)))

(defn weightv "Creates and returns a new weight vector" [n]
  (array (mapv (fn [x](random-weight)) (range n))))

(defn weightm "Creates and returns a new weight matrix" [n m]
  (array (mapv (fn [x] (mapv (fn [y](random-weight))(range n))) (range m))))

(defn v-net-to-m-net [v-net]
  (map (fn [layer] (list (array (first layer))(array (second layer))))
       v-net))

;;;-----------------------------------------------------------------------
;;; Matrix Like Operations
;;;-----------------------------------------------------------------------

(defn sub-row [row offset length]
  (take length (drop offset row)))

;;;-----------------------------------------------------------------------

(defn sub-matrix [m [row-offset col-offset] [rows cols]]
  (mapv (fn [row](sub-row row col-offset cols)) (take rows (drop row-offset m))))

;;;-----------------------------------------------------------------------
;;; CONVOLUTIONAL NEURAL NETS
;;;-----------------------------------------------------------------------

(def filter-size 4)
(def slide 1)

(defn gen-cnn-unit [image [row-offset col-offset] filter-size]
  (let [m (sub-matrix image [row-offset col-offset] [filter-size filter-size])]
    
  nil)

;;;-----------------------------------------------------------------------
;;; Generate Convolutional Layer
;;;-----------------------------------------------------------------------

;;; <filter-size> i the size of the local receptive field
;;; <feature-maps> is the number of feature maps.

(defn gen-conv-layer [image-size filter-size feature-maps]
  [(weightv feature-maps)(weightm (* filter-size filter-size) feature-maps)])

;;;-----------------------------------------------------------------------
;;; Layer Outputs
;;;-----------------------------------------------------------------------

;;; Output vector O of layer k: Ok = ¦Ò (WkOk-1 + Bk)

(defn layer-outputs [inputs [bias weights]]
  ())

;;;-----------------------------------------------------------------------
;;; Network Ouputs
;;;-----------------------------------------------------------------------

(defn network-outputs [inputs net]
  (reduce (fn [inputs layer] (layer-outputs inputs layer)) inputs net))

;;;-----------------------------------------------------------------------
;;; Compute Network
;;;-----------------------------------------------------------------------

;;; Returns a vector of the outputs of each layer

(defn compute-network [inputs net]
  (reduce (fn [acc layer] (conj acc (layer-outputs (last acc) layer)))
          [inputs]
          net))

;;;-----------------------------------------------------------------------
;;; End of File
;;;-----------------------------------------------------------------------

