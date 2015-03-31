package main;

import java.awt.TextArea;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintStream;

import javax.swing.JOptionPane;
import javax.swing.SwingUtilities;

public class Processing {

	private static void redirectSystemStreams(final TextArea textArea) {
		OutputStream out = new OutputStream() {
			@Override
			public void write(int b) throws IOException {
				updateTextArea(String.valueOf((char) b), textArea);
			}

			@Override
			public void write(byte[] b, int off, int len) throws IOException {
				updateTextArea(new String(b, off, len), textArea);
			}

			@Override
			public void write(byte[] b) throws IOException {
				write(b, 0, b.length);
			}
		};

		System.setOut(new PrintStream(out, true));
		System.setErr(new PrintStream(out, true));
	}

	private static String os;

	private static void updateTextArea(final String text,
			final TextArea textArea) {
		SwingUtilities.invokeLater(new Runnable() {
			public void run() {
				textArea.append(text);
			}
		});
	}

	static String svmParam(String[] parameters) {
		return "./CPP_BIO" + " --C=" + parameters[0] + " --svm="
				+ parameters[1] + " --g=" + parameters[2] + " --Cstep="
				+ parameters[3] + " --gStep=" + parameters[4] + " --iter="
				+ parameters[5] + " --data=" + parameters[6] + " --data1="
				+ parameters[7] + " --classification=" + parameters[8]
				+ " --show-graph=" + parameters[9] + " --Cstart="
				+ parameters[10] + " --gStart=" + parameters[11]
				+ parameters[12];
	}

	static String annParam(String[] parameters) {
		return "./CPP_BIO" + " --ann=true" + " --svm=" + parameters[0]
				+ " --cascade=" + parameters[1] + " --num_layers="
				+ parameters[2] + " --num_neurons_hidden=" + parameters[3]
				+ " --max_epochs=" + parameters[4] + " --iter=" + parameters[5]
				+ " --data=" + parameters[6] + " --data1=" + parameters[7]
				+ " --classification=" + parameters[8] + " --show-graph="
				+ parameters[9] + " --max_neurons=" + parameters[10]
				+ " --desired_error=" + parameters[11];
	}

	public static void DoProcessing(String[] parameters, TextArea textArea,
			boolean isSVM) {
		// redirectSystemStreams(textArea);
		os = System.getProperty("os.name");

		String cmd;
		if (isSVM)
			cmd = svmParam(parameters);
		else {
			cmd = annParam(parameters);
		}

		if (!os.equals("Linux"))
			cmd = "dir";
		Runtime run = Runtime.getRuntime();

		Process pr;
		try {
			pr = run.exec(cmd);

			// pr.waitFor();

			// BufferedWriter bwr = new BufferedWriter(new
			// OutputStreamWriter( pr.getOutputStream()));
			BufferedReader buf = new BufferedReader(new InputStreamReader(
					pr.getInputStream()));
			// bwr.write("fah6Mi6g\n");
			String line = "";
			String rem = "";
			while ((line = buf.readLine()) != null) {
				System.out.println(line);
				rem += line + "\r\n";
				updateTextArea(line + "\r\n", textArea);
				// textArea.setText(rem);
			}

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			JOptionPane.showMessageDialog(null,
					"Program launch not successfull", "alert",
					JOptionPane.ERROR_MESSAGE);
		}

	}
}
