package org.deeplearning4j.optimize

import org.deeplearning4j.nn.api.Layer
import org.deeplearning4j.nn.conf.layers.RBM
import org.deeplearning4j.nn.layers.factory.LayerFactories
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4s.nn.conf.NeuralNetConf
import org.deeplearning4s.optimize.SolverAt
import org.scalatest.FlatSpec

class SolverAtTest extends FlatSpec {
  "SolverAt" should "create a solver to return optimizer without exception" in {
    // FIX these sample creation after developing scala bindings for nn.conf.layers and nn.api.layers.
    val conf = NeuralNetConf(new RBM.Builder(RBM.HiddenUnit.BINARY, RBM.VisibleUnit.BINARY, 1).build()).asJava
    val model: Layer = LayerFactories.getFactory(conf.getLayer()).create(conf);
    val listeners = Seq(new ScoreIterationListener())

    SolverAt(conf, model, listeners).getOptimizer
  }
}