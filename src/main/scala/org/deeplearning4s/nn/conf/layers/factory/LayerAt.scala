package org.deeplearning4s.nn.conf.layers.factory

import org.deeplearning4j.nn.api.Layer
import org.deeplearning4j.nn.layers.factory.LayerFactories
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4s.nn.conf.NeuralNetConf

import scala.collection.JavaConverters._

object CreateLayerAt {
  def apply(conf: NeuralNetConf, index: Int, numLayers: Int, iterationListeners: Seq[IterationListener]): Layer = {
    val jConf = conf.asJava
    LayerFactories.getFactory(jConf).create(jConf, index, numLayers, iterationListeners.asJava)
  }

  def apply(conf: NeuralNetConf): Layer = {
    val jConf = conf.asJava
    LayerFactories.getFactory(jConf).create(jConf)
  }

  def apply(conf: NeuralNetConf, index: Int, iterationListeners: Seq[IterationListener]): Layer = {
    val jConf = conf.asJava
    LayerFactories.getFactory(jConf).create(jConf, iterationListeners.asJava, index)
  }

}