package org.deeplearning4j.nn.conf.layers.factory

import org.deeplearning4j.nn.conf.layers.{RBM => jRBM}
import org.deeplearning4s.nn.conf.{Tanh, NeuralNetConf}
import org.deeplearning4s.nn.conf.layers.factory.CreateLayerAt
import org.deeplearning4s.nn.conf.layers.{AutoEncoder, RBM}
import org.scalatest.FlatSpec

class CreateLayerAtTest extends FlatSpec {
  "CreateLayerAt" should "create DL4j's nn.api.Layer instances corresponding to passed NeuralNetConf" in {

    val confRBM = NeuralNetConf(
      layer = RBM(
        hiddenUnit = jRBM.HiddenUnit.BINARY,
        visibleUnit = jRBM.VisibleUnit.BINARY,
        k = 1))

    CreateLayerAt(confRBM) match {
      case _: org.deeplearning4j.nn.layers.feedforward.rbm.RBM => assert(true)
      case _ => assert(false)
    }

    val confAE = NeuralNetConf(
      layer = AutoEncoder(
        activationFunction = Tanh,
        corruptionLevel = 0.1,
        sparsity = 0.2
      ))

    CreateLayerAt(confAE) match {
      case _: org.deeplearning4j.nn.layers.feedforward.autoencoder.AutoEncoder => assert(true)
      case _ => assert(false)
    }
  }
}