package org.deeplearning4j.nn.conf.layers

import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.distribution.{Distribution, NormalDistribution}
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4s.nn.conf.layers.{AutoEncoder => sAutoEncoder}
import org.deeplearning4s.nn.conf.{ActivationFunction, Tanh}
import org.scalatest.FlatSpec

class AutoEncoderTest extends FlatSpec {
  "AutoEncoder" should "set each property to org.deeplearning4j.nn.conf.layers.AutoEncoder instance correctly" in {
    val corruptionLevel: Double = 0.1
    val sparsity: Double = 0.2
    val nIn: Int = 10
    val nOut: Int = 10
    val activationFunction: ActivationFunction = Tanh
    val weightInit: WeightInit = WeightInit.VI
    val dist: Distribution = new NormalDistribution(1e-3, 1)
    val updater: Updater = Updater.NONE
    val dropOut: Double = 0.2

    val rbm = sAutoEncoder(
      corruptionLevel = corruptionLevel,
      sparsity = sparsity,
      nIn = 10,
      nOut = 10,
      activationFunction = activationFunction,
      weightInit = weightInit,
      dist = dist,
      updater = updater,
      dropOut = dropOut
    ).asJava

    assert(rbm.corruptionLevel == corruptionLevel)
    assert(rbm.sparsity == sparsity)
    assert(rbm.nIn == nIn)
    assert(rbm.nOut == nOut)
    assert(rbm.activationFunction == activationFunction.name)
    assert(rbm.weightInit == weightInit)
    assert(rbm.dist == dist)
    assert(rbm.updater == updater)
    assert(rbm.dropOut == dropOut)
  }
}