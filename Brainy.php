<?php

include_once "Matrix.php";

class Brainy extends Matrix {
  private $learning_rate = 0.01;
  private $activation_fun = 'relu';


  /**
   * Set you neural network with the learning rate anf the activation function you prefer
   *
   * @param numeric $learning_rate  choose your learning rate
   * @param string $activation_fun  choose your activation funciton: RELU or SIGMOID or TANH
   */
  public function __construct( $learning_rate , $activation_fun ) {
    
    $activation_fun = strtolower( $activation_fun );
    if ( !is_numeric( $learning_rate ) ) {
      throw new Exception('The learning rate is not numeric');
    }

    if ( !in_array( $activation_fun , ['relu', 'sigmoid', 'tanh'] ) ) {
      throw new Exception('The allowed activation funciton are: RELU, SIGMOID, TANH');
    }

    $this->learning_rate = $learning_rate;
    $this->activation_fun = $activation_fun;
  }


  public function initialise( $input , $outputs , $hidden_layer_neurons ) {

    $input_neurons = count( $input[0] );
    $output_neurons = count( $outputs[0] );

    $weight['hidden_layer'] = array();
    $bias['hidden_layer'] = array();

    foreach($hidden_layer_neurons as $index => $size) {

      if( $index == array_key_first( $hidden_layer_neurons ) ) {

        // ------------------------------- hidden layer 1 ------------------------------------//
        // getting the W1 weights random matrix (layer between input and the hidden layer) with size 2 x $hidden_layer_neurons
        $weight['hidden_layer'][ $index ] = $this->getRandMatrix( $input_neurons , $hidden_layer_neurons[ $index ] );
      }
      else {

        // ------------------------------- hidden layer 2 ------------------------------------//
        // getting the W1 weights random matrix (layer between input and the hidden layer) with size 2 x $hidden_layer_neurons
        // set matrix row to size of previous hidden layers, and the columb to the size of this hidden layer
        $weight['hidden_layer'][ $index ] = $this->getRandMatrix( $hidden_layer_neurons[ $index-1 ] , $hidden_layer_neurons[ $index ] );
      }

      // getting the B1 bies random vector with size $hidden_layer_neurons
      $bias['hidden_layer'][ $index ] = $this->getRandMatrix( $hidden_layer_neurons[ $index ] , 1 );
    }

    // ------------------------------- output layer ------------------------------------//

    // getting the W2 weights random vector (layer between hidden layer and output) with size $hidden_layer_neurons x 1
    $weight['output'] = $this->getRandMatrix( end( $hidden_layer_neurons ) , $output_neurons );

    // getting the B2 bies random vector. The size is 1x1 because there is only one output neuron
    $bias['output'] =  $this->getRandMatrix( $output_neurons , 1 );

    return ['bias' => $bias
          ,'weight' => $weight
          ,];
  }

  /**
   * Forward propagation: it gets the output (matrix A)
   *
   * @param array $input  is the input vector
   * @param array $w1     the weights matrix (between input and hidden layer)
   * @param array $b1     the bias vector (between input and hidden layer)
   * @param array $w2     the weights matrix (between the hidden layer and the output)
   * @param array $b2     the bias vector (between the hidden layer and the output)
   * @return array        it returns the matrix A, which is the final output, and Z, which is the output of the hidden layer
   */

  public function forward_hidden_layer( $input , $weight , $bias )  {

    // Z = TANH( W1°X + B1 )    --> TANH or RELU or SIGMOID
    $dot = $this->matrixDotProduct( $this->matrixTranspose( $weight ), $input );
    $sum = $this->matrixSum( $dot , $bias );
    $z = $this->matrixOperation( $this->activation_fun , $sum );
    return $z;

  }
  public function forward( $input, $weight , $bias ) {

    $z = array ();

    foreach( ( $weight['hidden_layer'] ) as $index => $weights ) {

      // if this is the first hidden layer, set input as the input
      if( $index == array_key_first( $weight ) ) {
        $z[ $index ] = $this->forward_hidden_layer( $input, $weight['hidden_layer'][ $index ], $bias['hidden_layer'][ $index ] );

      }
      else {
        // else set the previous hidden layers output as this hidden layers input
        $z[ $index ] = $this->forward_hidden_layer( end( $z ) , $weight['hidden_layer'][ $index ], $bias['hidden_layer'][ $index ] );
      }
      // Z = TANH( W1°X + B1 )    --> TANH or RELU or SIGMOID
    }

    // use the last hidden layers output as the output layers input
    // A = SOFTMAX( W2°Z + B2 )
    $dot = $this->matrixDotProduct( $this->matrixTranspose( $weight['output'] ), end( $z ) );
    $sum = $this->matrixSum( $dot , $bias['output'] );
    $a = $this->matrixSoftmax( $sum );
    
    return [
       'output' => $a // output
      ,'hidden_layer' => $z // output of the hidden layer
    ];
  }



  /**
   * This is the function that corrects the weights propapating back the errors.
   * The function is using the gradient descent in order to find the minimum.
   *
   * @param array $r      if the matrix from the forard matrix
   * @param array $input  is the input vector
   * @param array $output the expected output vector
   * @param array $w1     the weights matrix (between input and hidden layer)
   * @param array $b1     the bias vector (between input and hidden layer)
   * @param array $w2     the weights matrix (between the hidden layer and the output)
   * @param array $b2     the bias vector (between the hidden layer and the output)
   * @return array        matrices of the weights (W1 and W2) and bias (B1 and B2) with new values
   */
  public function backPropagation( $predicted , $input , $output , $weight , $bias ) {

    // print("<pre>".print_r($weight, true)."</pre>");

    // determine the error, compare the difference between the predicted out vs the expected output
    // this is (A - Ouput) wherever this is found it can be subsituted for difference
    $difference = $this->matrixSub( $predicted['output'] , $output );

    // last hidden layer activated output, difference (total error of the network), weight of the hidden layer, bias of the hidden layer
    $last_layer = $this->last_layer_correction( end( $predicted['hidden_layer'] ), $difference , $weight['output'] , $bias['output'] );

    $index = count( $predicted['hidden_layer'] );

    while( $index ) {
      
      $key = $index - 1;

      if( $key == array_key_first( $predicted['hidden_layer'] ) ) {

        // if hidden layer is the first hidden layer set input to input
        // hidden layer is the last hidden layer set previous weights to the weight of output weights
        if( $key == array_key_last( $predicted['hidden_layer'] ) ) {

          $hidden_layer[ $key ] = $this->hidden_layer_correction( $input , $predicted['hidden_layer'][ $key ] , $difference, $weight['hidden_layer'][ $key ] , $bias['hidden_layer'][ $key ] , $weight['output'] );
        }
        else {

          // else set previous weights to the weight of the hidden layer that came after this hidden layer
          $hidden_layer[ $key ] = $this->hidden_layer_correction( $input, $predicted['hidden_layer'][ $key ] , $difference , $weight['hidden_layer'][ $key ] , $bias['hidden_layer'][ $key ] , $weight['hidden_layer'][ $key+1 ] , end( $hidden_layer) ['drivative'] );
        }
      }
      else {

        // else set input to the output of the hidden layer that came before it
        // input to the hidden layer, output of the hidden layer, difference (total error of the network), weight of the hidden layer, bias of the hidden layer, weight of the next layer, drivative of the previous hidden layer
        // hidden layer is the last hidden layer set previous weights to the weight of output weights
        if($key == array_key_last($predicted['hidden_layer'])) {
          $hidden_layer[ $key ] = $this->hidden_layer_correction( $predicted['hidden_layer'][ $key-1 ] , $predicted['hidden_layer'][ $key ] , $difference, $weight['hidden_layer'][ $key ] , $bias['hidden_layer'][ $key ] , $weight['output'] );
        }
        else {
          // else set previous weights to the weight of the hidden layer that came after this hidden layer
          $hidden_layer[ $key ] = $this->hidden_layer_correction( $predicted['hidden_layer'][ $key-1 ] , $predicted['hidden_layer'][ $key ] , $difference, $weight['hidden_layer'][ $key ] , $bias['hidden_layer'][ $key ] , $weight['hidden_layer'][ $key+1 ] , end( $hidden_layer )['drivative'] );
        }
      }
      --$index;
    }

    // // input to the hidden layer, output of the hidden layer, difference (total error of the network), weight of the hidden layer, bias of the hidden layer, weight of the next layer
    // $hidden_layer = $this->hidden_layer_correction($input, $predicted['hidden_layer'][0], $difference, $weight['hidden_layer'][0], $bias['hidden_layer'][0], $weight['output']);

    //https://stackoverflow.com/questions/5422242/array-map-not-working-in-classes

    return [
      'hidden_layer' => [ 'weight' => array_values( array_reverse( array_map( array( $this, "return_weights" ), $hidden_layer ) ) ),
                         'bias' => array_values( array_reverse( array_map( array( $this, "return_bias" ), $hidden_layer ) ) ) ],
      'output' => [ 'weight' => $last_layer['weights'],
                    'bias' => $last_layer['bias']],
    ];
  }

  public function return_weights( $layer ) {
    return $layer['weights'];
  }

  public function return_bias( $layer ) {
    return $layer['bias'];
  }
  public function last_layer_correction( $last_activated_output , $difference , $weights , $bias ) {

    // -------------------------- this is for the last layer --------------------------//
    // correcting the weights, where ( weight = weights - ( learning_rate * (A - Out) ° Z^T )^T )
    // ( weights = weights - ( learning_rate * ( difference ) ° Z^T )^T )
    // tranpose z_transform output of the hidden layer, using the last hidden layer activated output
    $z_transform = $this->matrixTranspose( $last_activated_output );

    // calculate the dot product z transposed and difference
    $product = $this->matrixDotProduct( $difference , $z_transform );
    $correction = $this->matrixTimesValue( $product , $this->learning_rate );

    // subsitute the orginal weight from the new correction weight
    $weights = $this->matrixSub( $weights, $this->matrixTranspose( $correction ) );

    // correcting the bias where ( bias = bias - learning_rate * (A - Out) )
    // ( bias = bias - learning_rate * ( difference ) )
    $correction = $this->matrixTimesValue( $difference , $this->learning_rate );

    // subsitute the orginal bias from the new correction bias
    $bias = $this->matrixSub( $bias , $correction );

    return [
      'weights' => $weights
     ,'bias' => $bias
   ];

  }

  // using the chain rule to calculate the corrections new weights and biases 
  // using the drivative of the of the output of that layer
  // the input to this layer, the activated output of this this layer, and the systems error
  public function hidden_layer_correction( $input , $output , $difference , $weights , $bias , $weight_previous , $dZ = null ) {

    // -------------------------- this is for the first layer of the neaural network -------------------------- //
    // calculating the derivative depending on the activate funciton, where ( dZ = (weights[n-1] ° (A - Out)) * DERIVATIVE(Z) ) 
    // where ( dZ = (weights[n-1] ° ( difference )) * DERIVATIVE(Z) ) 

    // Z == the output of the hidden layer
    // weights ==  the weights of the hidden layer
    // weights[n-1] == the weights of the previous layer

    // calculate the chain rule drivative 
    if( $dZ == null ) {
      $z_drivative = $this->derivative( $output );
      $product = $this->matrixDotProduct( $weight_previous , $difference);
      $dZ = $this->matrixProductValueByValue( $product ,  $z_drivative );
    }
    else {
      $z_drivative = $this->derivative($output);
      $product = $this->matrixDotProduct( $weight_previous , $dZ);
      $dZ = $this->matrixProductValueByValue( $product ,  $z_drivative);
    }

    // correctiong the weights, where (weights = weights - ( learning_rate * (dZ ° In^T) )^T)
    $product = $this->matrixDotProduct( $dZ , $this->matrixTranspose( $input ) );
    $correction = $this->matrixTimesValue( $product , $this->learning_rate );

    // subsitute the orginal weight from the new correction weight
    $weights = $this->matrixSub( $weights, $this->matrixTranspose( $correction ) );

    // correcting the bias bias, where (bias = bias - learning_rate * dZ)
    // bias == the bias of this hidden layer
    $correction = $this->matrixTimesValue( $dZ , $this->learning_rate );

    // subsitute the orginal bias from the new correction bias
    $bias = $this->matrixSub( $bias, $correction );

    return [
      'weights' => $weights
     ,'bias' => $bias
     ,'drivative'=> $dZ
     ,
   ];
  }

  // return z_drivative depending upon the activation function being used
  public function derivative( $Z ) {
    if ( $this->activation_fun == 'tanh' ) {
      $z_drivative = $this->tanhDerivative( $Z );
    }
    else if ( $this->activation_fun == 'sigmoid' ) {
      $z_drivative = $this->sigmoidDerivative( $Z );
    }
    else if ( $this->activation_fun == 'relu' ) {
      $z_drivative = $this->reluDerivate( $Z );
    }

    return $z_drivative;
  }

  /**
   * It calculates the Sigmoid derivative of a given matrix
   * d/dX sigm = sigm (1 - sigm)
   *
   * @param array $matrix   the matrix where you want to calculate the derivative
   * @return array          the final matrix
   */
  public function sigmoidDerivative( $z ) {
    $z_2 = $this->matrixProductValueByValue( $z , $z );
    return $this->matrixSub( $z , $z_2 );
  }


  /**
   * It calculates the Relu derivative of a given matrix
   * d/dX relu =  if (x > 0) then 1 else 0
   *
   * @param array $matrix   the matrix where you want to calculate the derivative
   * @return array          the final matrix
   */
  public function reluDerivate( $z ) {

    $relu_der = [];

    foreach( $z as $row_num => $row ) {
      foreach( $row as $col_num => $val ) {
        $relu_der[ $row_num ][ $col_num ] = ( $val > 0 ) ? 1 : 0;
      }
    }

    return $relu_der;
  }


  /**
   * It calculates the Hyperbolic Tangent derivative of a given matrix
   * d/dX tanh = (1-(tanh)^2)
   *
   * @param array $matrix   the matrix where you want to calculate the derivative
   * @return array          the final matrix
   */
  public function tanhDerivative( $matrix ) {
    $matrix_square = $this->matrixProductValueByValue( $matrix , $matrix );
    $matrix_neg = $this->matrixTimesValue( $matrix_square , -1);
    return $this->matrixSumValue( $matrix_neg , 1);
  }


}
