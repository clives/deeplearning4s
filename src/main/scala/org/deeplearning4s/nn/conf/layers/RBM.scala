package org.deeplearning4s.nn.conf.layers

import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.distribution.Distribution
import org.deeplearning4j.nn.conf.layers.{RBM => jRBM}
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4s.nn.conf.ActivationFunction

case class RBM(
                hiddenUnit: jRBM.HiddenUnit = jRBM.HiddenUnit.BINARY,
                visibleUnit: jRBM.VisibleUnit = jRBM.VisibleUnit.BINARY,
                k: Int = Int.MinValue,
                override val nIn: Int = Int.MinValue,
                override val nOut: Int = Int.MinValue,
                override val activationFunction: ActivationFunction = null,
                override val weightInit: WeightInit = null,
                override val dist: Distribution = null,
                override val updater: Updater = null,
                override val dropOut: Double = Double.NaN
                ) extends BasePretrainNetwork {

  def asJava = new jRBM.Builder()
    .activation(if (activationFunction == null) null else activationFunction.name)
    .weightInit(weightInit)
    .dist(dist)
    .updater(updater)
    .dropOut(dropOut)
    .nIn(nIn)
    .nOut(nOut)
    .hiddenUnit(hiddenUnit)
    .visibleUnit(visibleUnit)
    .k(k)
    .build()
}