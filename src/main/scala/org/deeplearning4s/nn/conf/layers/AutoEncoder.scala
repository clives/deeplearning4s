package org.deeplearning4s.nn.conf.layers

import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.distribution.Distribution
import org.deeplearning4j.nn.conf.layers.AutoEncoder.Builder
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4s.nn.conf.ActivationFunction

case class AutoEncoder(
                        corruptionLevel: Double = Double.NaN,
                        sparsity: Double = Double.NaN,
                        override val nIn: Int = Int.MinValue,
                        override val nOut: Int = Int.MinValue,
                        override val activationFunction: ActivationFunction = null,
                        override val weightInit: WeightInit = null,
                        override val dist: Distribution = null,
                        override val dropOut: Double = Double.NaN,
                        override val updater: Updater = null
                        ) extends BasePretrainNetwork {
  def asJava = new Builder(corruptionLevel)
    .activation(activationFunction.name)
    .weightInit(weightInit)
    .dist(dist)
    .dropOut(dropOut)
    .updater(updater)
    .nIn(nIn)
    .nOut(nOut)
    .sparsity(sparsity).build()
}
