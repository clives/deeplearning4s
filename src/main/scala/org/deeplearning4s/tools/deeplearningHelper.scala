package org.deeplearning4s.tools
import org.deeplearning4j.optimize.api.IterationListener
import scala.collection.JavaConverters._
import org.deeplearning4s.nn.conf.ActivationFunction
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.Layer
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder
import akka.dispatch.Foreach
import org.deeplearning4j.nn.conf.layers.OutputLayer


/*
 * implicit conversion to simply the use of the library from scala	
 */
object deeplearningHelper {

  //Seq[IterationListener](new ScoreIterationListener(listenerFreq)).asJava
  
	implicit def listenerToListListener( listener: IterationListener )= {
	  Seq[IterationListener](listener).asJava
	}

	implicit def ActivationFunctiontoString( activ: ActivationFunction ):String={
	  activ.name
	}
	
	case class configInOut( in: Int, out: Int)
	
	implicit class addInOut( val in: Int) extends AnyVal{
	  def -> (out: Int) = configInOut( in, out)
	}
	
	def DenseLayerBuilder( inout: configInOut ) = new DenseLayer.Builder().nIn(inout.in).nOut(inout.out)
	
	def OutputLayerBuilder = new OutputLayer.Builder()
	
	//construct layers using
	//  layer1 |> layer2 |> layer3
	
	implicit class addLayer(val a: Layer) extends AnyVal {
		def |>(b: Layer):List[Layer] = 
		  List( a, b)
	}
	
	implicit class addListLayer(val a: List[Layer]) extends AnyVal {
		def |>(b: Layer):List[Layer] = 
		  a :+ b
	}
	
	implicit class addListBuilder( val a: ListBuilder ) extends AnyVal {
		def setLayers( layers: List[Layer]):ListBuilder ={
		  layers.zipWithIndex.foreach{
		    case (layer, index) =>
		      a.layer(index , layer)
		  }
		  a
		}
	}
	
	
//	implicit def addFunctionActivationFunction(ourString: DenseLayer.Builder)=new{ 
//	  def activation(activ: ActivationFunction )={
//	    
//	       ourString.activation(activ.name)
//           ourString  
//	  } 
//	}
}