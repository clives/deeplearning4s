package org.deeplearning4s.nn.conf

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.Builder
import org.deeplearning4j.nn.conf.distribution.{Distribution, NormalDistribution}
import org.deeplearning4j.nn.conf.layers.{Layer, RBM, SubsamplingLayer}
import org.deeplearning4j.nn.conf.stepfunctions.StepFunction
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4s.nn.conf.LossFunction.{Custom, RECONSTRUCTION_CROSSENTROPY}
import org.nd4j.linalg.convolution.Convolution

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
                         lossFunction: LossFunction = RECONSTRUCTION_CROSSENTROPY,
                         constrainGradientToUnitNorm: Boolean = false,
                         seed: Long = System.currentTimeMillis(),
                         dist: Distribution = new NormalDistribution(1e-3, 1),
                         nIn: Int = 0,
                         nOut: Int = 0,
                         activationFunction: ActivationFunction = Sigmoid,
                         visibleUnit: RBM.VisibleUnit = RBM.VisibleUnit.BINARY,
                         hiddenUnit: RBM.HiddenUnit = RBM.HiddenUnit.BINARY,
                         weightShape: Array[Int] = null,
                         kernelSize: Array[Int] = Array(2, 2),
                         stride: Array[Int] = Array(2, 2),
                         padding: Array[Int] = Array(0, 0),
                         batchSize: Int = 100,
                         numLineSearchIterations: Int = 5,
                         maxNumLineSearchIterations: Int = 5,
                         minimize: Boolean = false,
                         convolutionType: Convolution.Type = Convolution.Type.VALID,
                         poolingType: SubsamplingLayer.PoolingType = SubsamplingLayer.PoolingType.MAX,
                         l1: Double = 0.0,
                         rmsDecay: Double = 0f,
                         stepFunction: StepFunction = null,
                         useDropConnect: Boolean = false,
                         rho: Double = 0D,
                         updater: Updater = Updater.NONE,
                         miniBatch: Boolean = false
                          ) {
  def asJava: NeuralNetConfiguration = {
    val ma = momentumAfter.map { case (i, d) => Integer.valueOf(i) -> (d: java.lang.Double) }.toMap.asJava
    val customLossFunction = lossFunction match {
      case c: Custom => c.name
      case _ => null
    }

    val conf = new Builder()
      .layer(layer)
      .sparsity(sparsity)
      .useAdaGrad(useAdaGrad)
      .learningRate(learningRate)
      .k(k)
      .corruptionLevel(corruptionLevel)
      .iterations(numIterations)
      .momentum(momentum)
      .l2(l2)
      .regularization(useRegularization)
      .momentumAfter(ma)
      .resetAdaGradIterations(resetAdaGradIterations)
      .dropOut(dropOut)
      .applySparsity(applySparsity)
      .weightInit(weightInit)
      .optimizationAlgo(optimizationAlgo)
      .lossFunction(lossFunction.value)
      .customLossFunction(customLossFunction)
      .constrainGradientToUnitNorm(constrainGradientToUnitNorm)
      .seed(seed)
      .dist(dist)
      .nIn(nIn)
      .nOut(nOut)
      .activationFunction(activationFunction.name)
      .visibleUnit(visibleUnit)
      .hiddenUnit(hiddenUnit)
      .weightShape(weightShape)
      .kernelSize(kernelSize: _*)
      .stride(stride)
      .padding(padding)
      .batchSize(batchSize)
      .numLineSearchIterations(numIterations)
      .maxNumLineSearchIterations(maxNumLineSearchIterations)
      .minimize(minimize)
      .convolutionType(convolutionType)
      .poolingType(poolingType)
      .l1(l1)
      .rmsDecay(rmsDecay)
      .stepFunction(stepFunction)
      .useDropConnect(useDropConnect)
      .rho(rho)
      .updater(updater)
      .miniBatch(miniBatch)
      .build()

    conf
  }
}
