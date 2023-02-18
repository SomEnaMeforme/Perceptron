using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Diagnostics;
using System.Drawing;
using System.Globalization;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;
using System.Transactions;
using ConsoleApp13;

namespace Cards_Game
{

    public class Neuron
    {
        public double[] weights;
        public double bias;

        public Neuron(int weightsCount)
        {
            weights = new double[weightsCount];

            var rnd = new Random();

            for (int i = 0; i < weightsCount; i++)
            {
                var param = rnd.Next(1, 2) % 2 == 1 ? 1 : -1;
                weights[i] = rnd.NextDouble() * param;
                bias = rnd.NextDouble() * param;
            }
        }

        internal double GetResult(double[] input)
        {
            double result = 0;
            for(int i = 0; i < input.Length; i++)
            {
                result += weights[i] * input[i] + bias;
            }

            //var exp = Math.Exp(result); //sigmoid activ_function
            return Math.Tanh(result);
        }

        internal void PropagateError(double error, double output)
        {
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = weights[i] - error * (1-output*output);
                bias += output * (1 - output);
            }
        }
    }

    public class Layer
    {
        public List <Neuron> neurons = new List<Neuron>();

        public Layer(int neuronsCount, Layer prevLayer = null)
        {

            for (int i = 0; i < neuronsCount; i++)
                neurons.Add(new Neuron(prevLayer?.LayerNeuronsCount ?? Perceptron.inputLayerSize));

        }

        public int LayerNeuronsCount => neurons.Count;

        internal double[] GetResult(double[] input)
        {
            var result = new List<double>(neurons.Count);

            foreach (var neuron in neurons)
            {
                result.Add(neuron.GetResult(input));
            }

            return result.ToArray();
        }

        internal void PropagateError(double error, double output)
        {
            for(int i = 0; i < neurons.Count; i++)
            {

                neurons[i].PropagateError(error, output);
            }
        }
    }

    public class Perceptron 
    {

        List<Layer> Layers = new List<Layer>();
        int layersCount = 7;
        int neuronsCount = 25;
        public static int inputLayerSize = 5;
        public static int outputLayerSize = 1;

        public Perceptron()
        {
            var prevLayer = new Layer(inputLayerSize);
            Layers.Add(prevLayer);
            for (var layerNumb = 0; layerNumb < layersCount; layerNumb++)
            {
                var layer = new Layer(neuronsCount, prevLayer);
                prevLayer = layer;
                Layers.Add(layer);
            }

            //create output layer

            Layers.Add(new Layer(outputLayerSize, prevLayer));

        }
      
        public int[] CreateInputData(List<Card> playerHand, SuitsTypes tramp, List<Card> onDeck)
        {
            var result = new List<int>();
            result.Add((int)tramp);
            for (int i = 0; i < playerHand.Count; i++)
            {
                result.Add((int)playerHand[i].value);
                result.Add((int)playerHand[i].suit);
            }
            for (int i = 0; i < onDeck.Count; i++)
            {
                result.Add((int)onDeck[i].value);
                result.Add((int)onDeck[i].suit);
            }
            return result.ToArray();
        }

    

        public double[] GetResult(double[] input)
        {
            for(int i = 0; i < Layers.Count - 1; i++)
            {
                input = Layers[i].GetResult(input);

            }
            input = Layers[Layers.Count - 1].GetResult(input);
            return input;
        }

        internal void PropagateError(double[] error, Dictionary<Card, double> allScores, List<Card> cards)
        {
            var test = error.Average();
            for(int i = 0; i< cards.Count; i++)
            {
                foreach (var layer in Layers.AsEnumerable().Reverse())
                {
                    layer.PropagateError(test, allScores[cards[i]]);
                }
            }
 
        }
    }
}

