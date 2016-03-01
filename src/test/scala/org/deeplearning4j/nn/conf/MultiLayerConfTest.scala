package org.deeplearning4j.nn.conf

import org.deeplearning4j.nn.conf.NeuralNetConfiguration.Builder
import org.deeplearning4j.nn.conf.`override`.ConfOverride
import org.deeplearning4s.nn.conf.MultiLayerConf
import org.nd4j.linalg.api.ndarray.INDArray
import org.scalatest.FlatSpec

import scala.collection.JavaConverters._

class MultiLayerConfTest extends FlatSpec {
  "MultiLayerConf" should "set each property to MultiLayerConfiguration instance correctly" in {
    val hiddenLayerSizesV: Array[Int] = Array(1)
    val useDropConnectV: Boolean = true
    val pretrainV: Boolean = true
    val useRBMPropUpAsActivationsV: Boolean = true
    val dampingFactorV: Double = 1000D
    val backpropV: Boolean = true
    val inputPreProcessorsV: Map[Int, InputPreProcessor] = Map(1 -> new InputPreProcessor {
      def backprop(output: INDArray, miniBatchSize:Int): INDArray = ???

      def preProcess(input: INDArray,miniBatchSize:Int): INDArray = ???
    })
    
    val confsV: Seq[NeuralNetConfiguration] = List(new NeuralNetConfiguration())
    val confOverridesV: Map[Int, ConfOverride] = Map(1 -> new ConfOverride {
      override def overrideLayer(i: Int, builder: Builder): Unit = ???
    })

    val mlc = MultiLayerConf(hiddenLayerSizesV,
      useDropConnect = useDropConnectV,
      pretrain = pretrainV,
      useRBMPropUpAsActivations = useRBMPropUpAsActivationsV,
      dampingFactor = dampingFactorV,
      backprop = backpropV,
      inputPreProcessors = inputPreProcessorsV,
      //outputPostProcessors = preProcessorsV,
      confs = confsV,
      confOverrides = confOverridesV
    ).asJava

    assert(mlc.pretrain == pretrainV)
    assert(mlc.dampingFactor == dampingFactorV)
    assert(mlc.backprop == backpropV)
    assert(mlc.inputPreProcessors == inputPreProcessorsV.map { case (i, p) => Integer.valueOf(i) -> p }.asJava)  
    assert(mlc.confs == confsV.asJava)
  }
}
