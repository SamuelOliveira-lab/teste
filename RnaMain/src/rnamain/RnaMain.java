package rnamain;

import java.io.*;
import java.util.*;

public class RnaMain {

    public static double[][][] base;

    public static void main(String[] args) throws Exception {
        base = carregarBase("winequality-red.csv");
        int entradas = base[0][0].length;
        int saidas = base[0][1].length;
        Perceptron rna = new Perceptron(entradas, saidas);
        for (int epoca = 0; epoca < 10000; epoca++) {
            double erroAproximacaoEpoca = 0;
            double erroClassificacaoEpoca = 0;
            for (int amostra = 0; amostra < base.length; amostra++) {
                double[] X = base[amostra][0];
                double[] Y = base[amostra][1];
                double[] out = rna.treinar(X, Y);
                double erroAmostra = 0;
                for (int i = 0; i < Y.length; i++) {
                    erroAmostra += Math.abs(Y[i] - out[i]);
                }
                erroAproximacaoEpoca += erroAmostra; 
                double[] outBin = new double[out.length];
                for (int i = 0; i < out.length; i++) {
                    if (out[i] >= 0.5) {
                        outBin[i] = 1.0;
                    } else {
                        outBin[i] = 0.0;
                    }
                }
                int erroClassifAmostra = 0;
                for (int i = 0; i < Y.length; i++) {
                    if (Math.abs(Y[i] - outBin[i]) > 0) {
                        erroClassifAmostra = 1;
                        break;
                    }
                }
                erroClassificacaoEpoca += erroClassifAmostra;
            } 
            System.out.println((epoca + 1) + " - " + erroAproximacaoEpoca + " - " + erroClassificacaoEpoca);
        }
    } 

    public static double[][][] carregarBase(String caminho) throws Exception {
        List<double[][]> dados = new ArrayList<>();
        BufferedReader br = new BufferedReader(new FileReader(caminho));
        String linha = br.readLine(); 
        while ((linha = br.readLine()) != null) {
            String[] partes = linha.split(";");
            double[] entrada = new double[partes.length - 1];
            for (int i = 0; i < partes.length - 1; i++) {
                entrada[i] = Double.parseDouble(partes[i]);
            } 
            int qualidade = Integer.parseInt(partes[partes.length - 1]);
            double[] saida = new double[1];
            if (qualidade >= 6) {
                saida[0] = 1.0;
            } else {
                saida[0] = 0.0;
            }
            dados.add(new double[][]{entrada, saida});
        }
        br.close(); 
        normalizar(dados);

        double[][][] baseArr = new double[dados.size()][2][];
        for (int i = 0; i < dados.size(); i++) {
            baseArr[i][0] = dados.get(i)[0];
            baseArr[i][1] = dados.get(i)[1];
        }
        return baseArr;
    } 

    private static void normalizar(List<double[][]> dados) {
        int atributos = dados.get(0)[0].length;
        double[] min = new double[atributos];
        double[] max = new double[atributos];
        Arrays.fill(min, Double.POSITIVE_INFINITY);
        Arrays.fill(max, Double.NEGATIVE_INFINITY);
        for (double[][] d : dados) {
            for (int i = 0; i < atributos; i++) {
                min[i] = Math.min(min[i], d[0][i]);
                max[i] = Math.max(max[i], d[0][i]);
            }
        }
        for (double[][] d : dados) {
            for (int i = 0; i < atributos; i++) {
                if (max[i] > min[i]) {
                    d[0][i] = (d[0][i] - min[i]) / (max[i] - min[i]);
                } else {
                    d[0][i] = 0.0;
                }
            }
        }
    }
}
