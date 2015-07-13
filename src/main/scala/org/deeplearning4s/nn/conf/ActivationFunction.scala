package org.deeplearning4s.nn.conf

/*
  org.deeplearning4s.nn.conf.ActivationFunction stands for each implemented class of org.nd4j.linalg.api.activation.ActivationFunction
  @see http://nd4j.org/apidocs/org/nd4j/linalg/api/activation/ActivationFunction.html
 */
sealed class ActivationFunction(val name:String)
case object Sigmoid extends ActivationFunction("sigmoid")
case object Tanh extends ActivationFunction("tanh")
case object Softmax extends ActivationFunction("softmax")
case object Relu extends ActivationFunction("relu")
case object Exp extends ActivationFunction("exp")
case object HardTanh extends ActivationFunction("hardtanh")
case object Linear extends ActivationFunction("linear")
case object MaxOut extends ActivationFunction("maxout")
case object RectifiedLinear extends ActivationFunction("rectifiedlinear")
case object RoundedLinear extends ActivationFunction("roundedlinear")
