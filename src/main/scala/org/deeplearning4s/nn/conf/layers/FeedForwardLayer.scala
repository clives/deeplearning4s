package org.deeplearning4s.nn.conf.layers

trait FeedForwardLayer extends Layer {
  val nIn: Int = Int.MinValue
  val nOut: Int = Int.MinValue
}