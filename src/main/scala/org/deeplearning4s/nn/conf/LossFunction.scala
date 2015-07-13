package org.deeplearning4s.nn.conf

import org.nd4j.linalg.lossfunctions.LossFunctions.{LossFunction => LF}

/*
  LossFunction stands for org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction.
  CustomLossFunction stands for all implemented classes of org.nd4j.linalg.api.ops.Op.
 */

sealed class LossFunction(val value:LF)
object LossFunction {
  case object EXPLL extends LossFunction(LF.EXPLL)

  case object MCXENT extends LossFunction(LF.MCXENT)

  case object MSE extends LossFunction(LF.MSE)

  case object NEGATIVELOGLIKELIHOOD extends LossFunction(LF.NEGATIVELOGLIKELIHOOD)

  case object RECONSTRUCTION_CROSSENTROPY extends LossFunction(LF.RECONSTRUCTION_CROSSENTROPY)

  case object RMSE_XENT extends LossFunction(LF.RMSE_XENT)

  case object SQUARED_LOSS extends LossFunction(LF.SQUARED_LOSS)

  case object XENT extends LossFunction(LF.XENT)

  sealed class Custom(val name: String) extends LossFunction(LF.CUSTOM)

  case object Max extends Custom("Max")

  case object MulOp extends Custom("MulOp")

  case object Eps extends Custom("Eps")

  case object Negative extends Custom("Negative")

  case object EqualTo extends Custom("EqualTo")

  case object Round extends Custom("Round")

  case object CosineSimilarity extends Custom("CosineSimilarity")

  case object SoftMaxDerivative extends Custom("SoftMaxDerivative")

  case object EuclideanDistance extends Custom("EuclideanDistance")

  case object Abs extends Custom("Abs")

  case object ScalarLessThanOrEqual extends Custom("ScalarLessThanOrEqual")

  case object ScalarAdd extends Custom("ScalarAdd")

  case object Exp extends Custom("Exp")

  case object ScalarGreaterThan extends Custom("ScalarGreaterThan")

  case object Mean extends Custom("Mean")

  case object DivOp extends Custom("DivOp")

  case object Ceil extends Custom("Ceil")

  case object Prod extends Custom("Prod")

  case object MaxOut extends Custom("MaxOut")

  case object Cos extends Custom("Cos")

  case object RDivOp extends Custom("RDivOp")

  case object RSubOp extends Custom("RSubOp")

  case object ScalarMax extends Custom("ScalarMax")

  case object Dot extends Custom("Dot")

  case object Variance extends Custom("Variance")

  case object ScalarDivision extends Custom("ScalarDivision")

  case object HardTanhDerivative extends Custom("HardTanhDerivative")

  case object NotEqualTo extends Custom("NotEqualTo")

  case object SigmoidDerivative extends Custom("SigmoidDerivative")

  case object ScalarSubtraction extends Custom("ScalarSubtraction")

  case object Bias extends Custom("Bias")

  case object Sqrt extends Custom("Sqrt")

  case object Sum extends Custom("Sum")

  case object VectorIFFT extends Custom("VectorIFFT")

  case object ATan extends Custom("ATan")

  case object GreaterThan extends Custom("GreaterThan")

  case object ASin extends Custom("ASin")

  case object ScalarNotEquals extends Custom("ScalarNotEquals")

  case object ScalarMultiplication extends Custom("ScalarMultiplication")

  case object Identity extends Custom("Identity")

  case object StandardDeviation extends Custom("StandardDeviation")

  case object ScalarSetValue extends Custom("ScalarSetValue")

  case object HardTanh extends Custom("HardTanh")

  case object Stabilize extends Custom("Stabilize")

  case object GreaterThanOrEqual extends Custom("GreaterThanOrEqual")

  case object Floor extends Custom("Floor")

  case object NormMax extends Custom("NormMax")

  case object ScalarGreaterThanOrEqual extends Custom("ScalarGreaterThanOrEqual")

  case object Norm1 extends Custom("Norm1")

  case object Pow extends Custom("Pow")

  case object ScalarReverseSubtraction extends Custom("ScalarReverseSubtraction")

  case object Norm2 extends Custom("Norm2")

  case object ScalarLessThan extends Custom("ScalarLessThan")

  case object SetRange extends Custom("SetRange")

  case object LessThan extends Custom("LessThan")

  case object SubOp extends Custom("SubOp")

  case object LessThanOrEqual extends Custom("LessThanOrEqual")

  case object CopyOp extends Custom("CopyOp")

  case object SoftMax extends Custom("SoftMax")

  case object Sigmoid extends Custom("Sigmoid")

  case object RectifedLinear extends Custom("RectifedLinear")

  case object VectorFFT extends Custom("VectorFFT")

  case object Log extends Custom("Log")

  case object Tanh extends Custom("Tanh")

  case object OneMinus extends Custom("OneMinus")

  case object IAMax extends Custom("IAMax")

  case object ManhattanDistance extends Custom("ManhattanDistance")

  case object Sign extends Custom("Sign")

  case object ScalarEquals extends Custom("ScalarEquals")

  case object ACos extends Custom("ACos")

  case object LinearIndex extends Custom("LinearIndex")

  case object Ones extends Custom("Ones")

  case object AddOp extends Custom("AddOp")

  case object ScalarReverseDivision extends Custom("ScalarReverseDivision")

  case object Sin extends Custom("Sin")

  case object Min extends Custom("Min")

}