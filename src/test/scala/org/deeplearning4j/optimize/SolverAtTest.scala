package org.deeplearning4j.optimize

import org.deeplearning4j.nn.conf.layers.{RBM => jRBM}
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4s.nn.conf.NeuralNetConf
import org.deeplearning4s.nn.conf.layers.RBM
import org.deeplearning4s.nn.conf.layers.factory.CreateLayerAt
import org.deeplearning4s.optimize.SolverAt
import org.scalatest.FlatSpec

class SolverAtTest extends FlatSpec {
  "SolverAt" should "create a solver to return optimizer without exception" in {
    val conf = NeuralNetConf(RBM(jRBM.HiddenUnit.BINARY, jRBM.VisibleUnit.BINARY, 1))
    val model = CreateLayerAt(conf)
    val listeners = Seq(new ScoreIterationListener())

    SolverAt(conf, model, listeners).getOptimizer
  }
}