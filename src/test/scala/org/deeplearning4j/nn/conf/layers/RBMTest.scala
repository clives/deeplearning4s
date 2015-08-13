package org.deeplearning4j.nn.conf.layers

import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.distribution.{NormalDistribution, Distribution}
import org.deeplearning4j.nn.conf.layers.{RBM => jRBM}
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4s.nn.conf.{Tanh, ActivationFunction}
import org.deeplearning4s.nn.conf.layers.{RBM => sRBM}
import org.scalatest.FlatSpec

class RBMTest extends FlatSpec {
  "RBM" should "set each property to org.deeplearning4j.nn.conf.layers.RBM instance correctly" in {
    val hiddenUnit: jRBM.HiddenUnit = jRBM.HiddenUnit.BINARY
    val visibleUnit: jRBM.VisibleUnit = jRBM.VisibleUnit.BINARY
    val k: Int = 10
    val nIn: Int = 10
    val nOut: Int = 10
    val activationFunction: ActivationFunction = Tanh
    val weightInit: WeightInit = WeightInit.VI
    val dist: Distribution = new NormalDistribution(1e-3, 1)
    val updater: Updater = Updater.NONE
    val dropOut: Double = 0.2

    val rbm = sRBM(
      hiddenUnit = hiddenUnit,
      visibleUnit = visibleUnit,
      k = k,
      nIn = 10,
      nOut = 10,
      activationFunction = activationFunction,
      weightInit = weightInit,
      dist = dist,
      updater = updater,
      dropOut = dropOut
    ).asJava

    assert(rbm.hiddenUnit == hiddenUnit)
    assert(rbm.visibleUnit == visibleUnit)
    assert(rbm.k == k)
    assert(rbm.nIn == nIn)
    assert(rbm.nOut == nOut)
    assert(rbm.activationFunction == activationFunction.name)
    assert(rbm.weightInit == weightInit)
    assert(rbm.dist == dist)
    assert(rbm.updater == updater)
    assert(rbm.dropOut == dropOut)
  }
}
