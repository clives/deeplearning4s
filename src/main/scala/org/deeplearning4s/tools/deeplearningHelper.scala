package org.deeplearning4s.tools
import org.deeplearning4j.optimize.api.IterationListener
import scala.collection.JavaConverters._
import org.deeplearning4s.nn.conf.ActivationFunction


/*
 * implicit conversion to simply the use of the library from scala	
 */
object deeplearningHelper {

  //Seq[IterationListener](new ScoreIterationListener(listenerFreq)).asJava
  
	implicit def listenerToListListener( listener: IterationListener )= {
	  Seq[IterationListener](listener).asJava
	}
	
	/*would be better to add a new method to the builder to accept ActivationFunction
	 * as parameters*/
	implicit def ActivationFunctiontoString( activ: ActivationFunction ):String={
	  activ.name
	}
}