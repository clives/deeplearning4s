package org.deeplearning4s.nn.conf

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.distribution.{Distribution, NormalDistribution}
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, Layer, RBM}
import org.deeplearning4j.nn.conf.stepfunctions.{DefaultStepFunction, StepFunction}
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4s.nn.conf.LossFunction.{Custom, RECONSTRUCTION_CROSSENTROPY}
import scala.collection.JavaConverters._

case class NeuralNetConf(layer: Layer,
                         sparsity: Double = 1f,
                         useAdaGrad: Boolean = true,
                         learningRate: Double = 1e-1D,
                         k: Int = 1,
                         corruptionLevel: Double = 3e-1D,
                         numIterations: Int = 100,
                         momentum: Double = 0.5D,
                         l2: Double = 0f,
                         useRegularization: Boolean = false,
                         momentumAfter: Map[Int, Double] = Map.empty,
                         resetAdaGradIterations: Int = -1,
                         dropOut: Double = 0,
                         applySparsity: Boolean = false,
                         weightInit: WeightInit = WeightInit.VI,
                         optimizationAlgo: OptimizationAlgorithm = OptimizationAlgorithm.CONJUGATE_GRADIENT,
                         lossFunction:LossFunction = RECONSTRUCTION_CROSSENTROPY,
                         constrainGradientToUnitNorm: Boolean = false,
                         seed: Long = System.currentTimeMillis(),
                         dist: Distribution = new NormalDistribution(1e-3, 1),
                         nIn: Int = 0,
                         nOut: Int = 0,
                         activationFunction: ActivationFunction = Sigmoid,
                         visibleUnit: RBM.VisibleUnit = RBM.VisibleUnit.BINARY,
                         hiddenUnit: RBM.HiddenUnit = RBM.HiddenUnit.BINARY,
                         weightShape: Array[Int] = null,
                         filterSize: Array[Int] = Array(2, 2, 2, 2),
                         stride: Array[Int] = Array(2, 2),
                         featureMapSize: Array[Int] = Array(2, 2),
                         kernel: Int = 5,
                         batchSize: Int = 100,
                         numLineSearchIterations: Int = 100,
                         minimize: Boolean = false,
                         convolutionType: ConvolutionLayer.ConvolutionType = ConvolutionLayer.ConvolutionType.MAX,
                         l1: Double = 0.0,
                         rmsDecay: Double = 0f,
                         stepFunction: StepFunction = new DefaultStepFunction()
                          ) {
  def asJava: NeuralNetConfiguration = {
    val ma = momentumAfter.map { case (i, d) => Integer.valueOf(i) -> (d: java.lang.Double)}.toMap.asJava
    val customLossFunction = lossFunction match {
      case c:Custom => c.name
      case _  => null
    }
    val conf = new NeuralNetConfiguration(sparsity, useAdaGrad, learningRate, k,
      corruptionLevel, numIterations, momentum, l2, useRegularization, ma,
      resetAdaGradIterations, dropOut, applySparsity, weightInit, optimizationAlgo, lossFunction.value,
      constrainGradientToUnitNorm, null, seed,
      dist, nIn, nOut, activationFunction.name, visibleUnit, hiddenUnit, weightShape, filterSize, stride, featureMapSize, kernel
      , batchSize, numLineSearchIterations, minimize, layer, convolutionType, l1, customLossFunction)
    conf.setRmsDecay(rmsDecay)
    conf.setStepFunction(stepFunction)
    conf
  }
}
