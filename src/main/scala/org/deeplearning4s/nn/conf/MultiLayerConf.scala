package org.deeplearning4s.nn.conf

import org.deeplearning4j.nn.conf.MultiLayerConfiguration.Builder
import org.deeplearning4j.nn.conf._
import org.deeplearning4j.nn.conf.`override`.ConfOverride
import scala.collection.JavaConverters._

case class MultiLayerConf(
                           hiddenLayerSizes: Array[Int],
                           useDropConnect: Boolean = false,
                           pretrain: Boolean = false,
                           useRBMPropUpAsActivations: Boolean = false,
                           dampingFactor: Double = 100D,
                           backprop: Boolean = false,
                           inputPreProcessors: Map[Int, InputPreProcessor] = Map.empty,
                          // outputPostProcessors: Map[Int, OutputPostProcessor] = Map.empty,
                           confs: Seq[NeuralNetConfiguration] = Nil,
                           confOverrides: Map[Int, ConfOverride] = Map.empty
                           ) {
  def asJava: MultiLayerConfiguration = {
    val builder = new Builder()
      //.hiddenLayerSizes(hiddenLayerSizes: _*)
      //.useDropConnect(useDropConnect)
      .pretrain(pretrain)
     // .useRBMPropUpAsActivations(useRBMPropUpAsActivations)
      .dampingFactor(dampingFactor)
      .backprop(backprop)
      .inputPreProcessors(inputPreProcessors.map { case (i, p) => (int2Integer(i), p) }.toMap.asJava)
     // .outputPostProcessors(outputPostProcessors.map { case (i, p) => (int2Integer(i), p) }.toMap.asJava)
      .confs(confs.asJava)

    confOverrides.collect { case (i, co) =>
      builder.getConfOverrides().put(i, co)
    }
    builder.build()
  }
}
