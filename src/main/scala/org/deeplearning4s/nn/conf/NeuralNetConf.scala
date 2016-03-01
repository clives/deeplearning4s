package org.deeplearning4s.nn.conf

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.Builder
import org.deeplearning4j.nn.conf.distribution.{Distribution, NormalDistribution}
import org.deeplearning4j.nn.conf.layers.{RBM, SubsamplingLayer}
import org.deeplearning4j.nn.conf.stepfunctions.StepFunction
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4s.nn.conf.LossFunction.{Custom, RECONSTRUCTION_CROSSENTROPY}
import org.deeplearning4s.nn.conf.layers.Layer
import org.nd4j.linalg.convolution.Convolution

import scala.collection.JavaConverters._


//updated using:
//https://github.com/deeplearning4j/deeplearning4j/issues/618 
case class NeuralNetConf(
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
    miniBatch: Boolean = false )
 {
  
  def asJava: NeuralNetConfiguration = {
    val ma = momentumAfter.map { case (i, d) => Integer.valueOf(i) -> (d: java.lang.Double) }.toMap.asJava

       
    val conf = new Builder()
      .layer(layer.asJava)
      .learningRate(learningRate)
      .iterations(numIterations)
      .momentum(momentum)
      .l2(l2)
      .regularization(useRegularization)
      .momentumAfter(ma)
      .optimizationAlgo(optimizationAlgo)
      .constrainGradientToUnitNorm(constrainGradientToUnitNorm)
      .seed(seed)
      .timeSeriesLength(timeSeriesLength)
      .maxNumLineSearchIterations(maxNumLineSearchIterations)
      .minimize(minimize)
      .l1(l1)
      .rmsDecay(rmsDecay)
      .stepFunction(stepFunction)
      .useDropConnect(useDropConnect)
      .rho(rho)
      .miniBatch(miniBatch)
      .build()

    conf
  }
}
