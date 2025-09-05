package rnamain;

import java.util.Random;

public class Perceptron {

    private double[][] W;
    private int qtdIn, qtdOut;
    private double ni = 0.3; // taxa de aprendizado

    public Perceptron(int qtdIn, int qtdOut) {
        this.qtdIn = qtdIn;
        this.qtdOut = qtdOut;

        W = new double[qtdIn + 1][qtdOut];
        Random rand = new Random();
        for (int i = 0; i < qtdIn + 1; i++) {
            for (int j = 0; j < qtdOut; j++) {
                W[i][j] = rand.nextDouble() * 0.6 - 0.3;
            }
        }
    }

    public double[] treinar(double[] xIN, double[] Y) {
        double[] X = new double[xIN.length + 1];
        X[0] = 1.0; // bias
        for (int i = 0; i < xIN.length; i++) {
            X[i + 1] = xIN[i];
        }

        double[] out = new double[qtdOut];
        for (int j = 0; j < qtdOut; j++) {
            double u = 0;
            for (int i = 0; i < qtdIn + 1; i++) {
                u += X[i] * W[i][j];
            }
            out[j] = 1 / (1 + Math.exp(-u)); // sigmoide
        }

        for (int j = 0; j < qtdOut; j++) {
            for (int i = 0; i < qtdIn + 1; i++) {
                double delta = ni * (Y[j] - out[j]) * X[i];
                W[i][j] += delta;
            }
        }

        return out;
    }
}