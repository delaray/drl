
;;;-----------------------------------------------------------------------
;;; update-weights
;;;-----------------------------------------------------------------------

(defn show-update-weights-args  [inputs weights-matrix errors]
  (println "-------------------------------------------\n")
  (println "Update weights args:\n")
  (println "Inputs shape: " (shape inputs))
  (println "Inputs: ")
  (show-matrix inputs)
  (println "Weights shape: " (shape weights-matrix)) 
  (println "Weights: ")
  (show-matrix weights-matrix)
  (println "Errors shape: " (shape errors))
  (println "Errors: ")
  (show-vector errors))
