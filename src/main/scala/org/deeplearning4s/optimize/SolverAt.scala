package org.deeplearning4s.optimize

import org.deeplearning4j.nn.api.Model
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.optimize.Solver
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4s.nn.conf.NeuralNetConf

import scala.collection.JavaConverters._

object SolverAt {
      def apply(conf:NeuralNetConf,
                 model:Model,listeners:Seq[IterationListener] = Nil):Solver  =
        new Solver.Builder().configure(conf.asJava).model(model).listeners(listeners.asJava).build()
}
