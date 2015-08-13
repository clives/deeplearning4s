package org.deeplearning4s.nn.conf.layers

import org.deeplearning4j.nn.conf.layers.{Layer => jLayer}
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.distribution.Distribution
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4s.nn.conf.ActivationFunction

trait Layer {
  val activationFunction: ActivationFunction = null
  val weightInit: WeightInit = null
  val dist: Distribution = null
  val dropOut: Double = Double.NaN
  val updater: Updater = null

  def asJava: jLayer
}