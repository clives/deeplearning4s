package org.deeplearning4j.nn.conf

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.distribution.{Distribution, NormalDistribution}
import org.deeplearning4j.nn.conf.layers.{RBM => jRBM, SubsamplingLayer}
import org.deeplearning4j.nn.conf.stepfunctions.StepFunction
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4s.nn.conf.LossFunction.Max
import org.deeplearning4s.nn.conf.layers.RBM
import org.deeplearning4s.nn.conf.{ActivationFunction, NeuralNetConf, Sigmoid}
import org.nd4j.linalg.convolution.Convolution
import org.scalatest.FlatSpec

import scala.collection.JavaConverters._

class NeuralNetConfTest extends FlatSpec {
  "NeuralNetConf" should "set each property to MultiLayerConfiguration instance correctly" in {
    val k: Int = 2
    val layer = RBM(
      hiddenUnit = jRBM.HiddenUnit.BINARY,
      visibleUnit = jRBM.VisibleUnit.BINARY,
      k = k)
    val sparsity: Double = 2f
    val useAdaGrad: Boolean = false
    val learningRate: Double = 1e-2D
    val corruptionLevel: Double = 3e-2D
    val numIterations: Int = 200
    val momentum: Double = 1D
    val l2: Double = 1f
    val useRegularization: Boolean = true
    val momentumAfter: Map[Int, Double] = Map(1 -> 1.0D)
    val resetAdaGradIterations: Int = -2
    // This value must be overridden by Layer's dropOut parameters
    val dropOut: Double = 1.0D
    val applySparsity: Boolean = true
    val weightInit: WeightInit = WeightInit.UNIFORM
    val optimizationAlgo: OptimizationAlgorithm = OptimizationAlgorithm.HESSIAN_FREE
    val lossFunction = Max
    val constrainGradientToUnitNorm: Boolean = true
    val seed: Long = System.currentTimeMillis()
    val dist: Distribution = new NormalDistribution(1e-3, 1)
    val nIn: Int = 0
    val nOut: Int = 0
    val activationFunction: ActivationFunction = Sigmoid
    val visibleUnit: jRBM.VisibleUnit = jRBM.VisibleUnit.BINARY
    val hiddenUnit: jRBM.HiddenUnit = jRBM.HiddenUnit.BINARY
    val weightShape: Array[Int] = Array(4, 4)
    val kernelSize: Array[Int] = Array(2, 2)
    val timeSeriesLength: Int = 1
    val stride: Array[Int] = Array(2, 2)
    val padding: Array[Int] = Array(1, 1)
    val batchSize: Int = 200
    val maxNumLineSearchIterations: Int = 100
    val minimize: Boolean = true
    val convolutionType: Convolution.Type = Convolution.Type.VALID
    val poolingType: SubsamplingLayer.PoolingType = SubsamplingLayer.PoolingType.MAX
    val l1: Double = 1.0D
    val rmsDecay: Double = 1f
    val stepFunction: StepFunction = new StepFunction
    val useDropConnect: Boolean = false
    val rho: Double = 0D
    val updater: Updater = Updater.NONE
    val miniBatch: Boolean = false

    /*
        layer : Layer,
    learningRate:  Double = 1e-1D,
    numIterations: Int = 100,
    momentum:  Double = 0.5D,
    l2:  Double = 0f,
    useRegularization: Boolean = false,
    momentumAfter: Map[Int, Double] = Map.empty,
    optimizationAlgo: OptimizationAlgorithm = OptimizationAlgorithm.CONJUGATE_GRADIENT,
    constrainGradientToUnitNorm: Boolean = false,
    seed:Long = System.currentTimeMillis(),
    timeSeriesLength: Int = 1,
    maxNumLineSearchIterations: Int = 5,
    minimize: Boolean = false,
    l1: Double = 0.0,
    rmsDecay: Double = 0f,
    stepFunction: StepFunction = null,
    useDropConnect: Boolean = false,
    rho: Double = 0D,
    miniBatch: Boolean = false )*/
    
    
    val conf = NeuralNetConf(
      layer = layer,
      learningRate = learningRate,    
      numIterations = numIterations,
      momentum = momentum,
      l2 = l2,
      useRegularization = useRegularization,
      momentumAfter = momentumAfter,     
      optimizationAlgo = optimizationAlgo,     
      constrainGradientToUnitNorm = constrainGradientToUnitNorm,
      seed = seed,      
      timeSeriesLength = 1,     
      maxNumLineSearchIterations = maxNumLineSearchIterations,
      minimize = minimize,    
      l1 = l1,
      rmsDecay = rmsDecay,
      stepFunction = stepFunction,
      useDropConnect = useDropConnect,
      rho = rho,
      miniBatch = miniBatch
    ).asJava

    assert(conf.layer == layer.asJava)
    assert(conf.numIterations == numIterations)
    assert(conf.useRegularization == useRegularization)
    assert(conf.optimizationAlgo == optimizationAlgo)
    assert(conf.seed == seed)
    assert(conf.maxNumLineSearchIterations == maxNumLineSearchIterations)
    assert(conf.minimize == minimize)
    assert(conf.stepFunction == stepFunction)
    assert(conf.useDropConnect == useDropConnect)
    assert(conf.miniBatch == miniBatch)
  }
}
