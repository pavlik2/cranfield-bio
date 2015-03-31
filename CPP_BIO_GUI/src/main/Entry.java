package main;

import java.awt.BorderLayout;
import java.awt.EventQueue;
import java.awt.TextArea;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;

import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.JSpinner;
import javax.swing.JTextField;
import javax.swing.SpinnerNumberModel;
import javax.swing.SwingUtilities;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.filechooser.FileFilter;

public class Entry {

	private JFrame frmCppbioGuic;
	private JTextField txtTrainDataFile;
	private JTextField txtClassDataFile;
	private final TextArea textArea = new TextArea();
	protected boolean sel1 = false;
	protected boolean sel2 = false;
	private final JRadioButton rdbtnNewRadioButton = new JRadioButton(
			"Support vector machines");
	private final JRadioButton rdbtnNewRadioButton_1 = new JRadioButton(
			"Nueral networks");
	private final JCheckBox chckbxCascadetraining = new JCheckBox(
			"CascadeTraining");
	private final JCheckBox chckbxGpuSupport = new JCheckBox("GPU support");

	/**
	 * Launch the application.
	 */
	public static void main(String[] args) {
		EventQueue.invokeLater(new Runnable() {
			public void run() {
				try {
					Entry window = new Entry();
					window.frmCppbioGuic.setVisible(true);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		});
	}

	/**
	 * Create the application.
	 */
	public Entry() {
		initialize();
	}

	public File choose() {

		// Create a file chooser
		final JFileChooser fc = new JFileChooser();
		fc.addChoosableFileFilter(new FileFilter() {

			@Override
			public String getDescription() {

				return "*.csv";
			}

			@Override
			public boolean accept(File arg0) {
				if (arg0.isDirectory()
						|| arg0.getAbsolutePath().endsWith(".csv"))
					return true;
				else
					return false;
			}
		});

		// In response to a button click:
		int returnVal = fc.showOpenDialog(null);

		if (returnVal == JFileChooser.APPROVE_OPTION)

			return fc.getSelectedFile();
		else
			return null;

	}

	/**
	 * Initialize the contents of the frame.
	 */
	private void initialize() {
		frmCppbioGuic = new JFrame();
		frmCppbioGuic.setTitle("CPP_BIO GUI (C) Pavel Kartashev");
		frmCppbioGuic.setBounds(100, 100, 687, 450);
		frmCppbioGuic.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

		JPanel panel = new JPanel();
		frmCppbioGuic.getContentPane().add(panel, BorderLayout.CENTER);
		panel.setLayout(null);

		JButton btnLaunch = new JButton("Launch");

		btnLaunch.setBounds(556, 384, 117, 25);
		panel.add(btnLaunch);

		final JLabel lblCparameter = new JLabel("C -parameter");
		lblCparameter.setBounds(12, 64, 129, 15);
		panel.add(lblCparameter);

		final JLabel lblGammaParameter = new JLabel("gamma parameter");
		lblGammaParameter.setBounds(12, 118, 177, 15);
		panel.add(lblGammaParameter);

		final JRadioButton rdbtnClassification = new JRadioButton(
				"Classification");

		rdbtnClassification.setSelected(true);
		rdbtnClassification.setBounds(556, 0, 149, 23);
		panel.add(rdbtnClassification);

		final JRadioButton rdbtnRegression = new JRadioButton("Regression");
		rdbtnRegression.setBounds(554, 34, 149, 23);
		panel.add(rdbtnRegression);

		rdbtnRegression.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent arg0) {
				if (rdbtnRegression.isSelected())
					rdbtnClassification.setSelected(false);
			}
		});

		rdbtnClassification.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent arg0) {
				if (rdbtnClassification.isSelected())
					rdbtnRegression.setSelected(false);
			}
		});
		final JSpinner spinner = new JSpinner(new SpinnerNumberModel(1, 1, 50,
				1));
		spinner.setBounds(233, 62, 51, 20);
		panel.add(spinner);

		final JLabel lblFrom = new JLabel("From");
		lblFrom.setBounds(233, 12, 70, 15);
		panel.add(lblFrom);

		final JLabel lblTo = new JLabel("To");
		lblTo.setBounds(344, 12, 70, 15);
		panel.add(lblTo);

		final JSpinner spinner_1 = new JSpinner();
		spinner_1.setModel(new SpinnerNumberModel(new Integer(50), null, null,
				new Integer(1)));
		spinner_1.setBounds(344, 62, 51, 20);
		panel.add(spinner_1);

		final JSpinner spinner_2 = new JSpinner();
		spinner_2.setModel(new SpinnerNumberModel(0.1, 0.1, 10, 0.1));
		spinner_2.setBounds(233, 116, 51, 20);
		panel.add(spinner_2);

		final JSpinner spinner_3 = new JSpinner();
		spinner_3.setModel(new SpinnerNumberModel(10.0, 0.1, 100.0, 0.1));
		spinner_3.setBounds(344, 116, 51, 20);
		panel.add(spinner_3);

		final JLabel lblSvmCount = new JLabel("SVM Count");
		lblSvmCount.setBounds(12, 163, 129, 15);
		panel.add(lblSvmCount);

		final JSpinner spinner_4 = new JSpinner();
		spinner_4.setModel(new SpinnerNumberModel(new Integer(200), null, null,
				new Integer(1)));
		spinner_4.setBounds(159, 161, 63, 20);
		panel.add(spinner_4);

		JLabel lblIterations = new JLabel("Iterations");
		lblIterations.setBounds(12, 208, 70, 15);
		panel.add(lblIterations);

		final JSpinner spinner_5 = new JSpinner();
		spinner_5.setModel(new SpinnerNumberModel(new Integer(1), null, null,
				new Integer(1)));
		spinner_5.setBounds(194, 206, 28, 20);
		panel.add(spinner_5);

		final JLabel lblStep = new JLabel("Step");
		lblStep.setBounds(423, 64, 129, 15);
		panel.add(lblStep);

		final JLabel label = new JLabel("Step");
		label.setBounds(423, 116, 129, 17);
		panel.add(label);

		final JSpinner spinner_6 = new JSpinner();
		spinner_6.addChangeListener(new ChangeListener() {

			@Override
			public void stateChanged(ChangeEvent e) {
				// TODO Auto-generated method stub

			}
		});
		spinner_6.setModel(new SpinnerNumberModel(new Integer(1),
				new Integer(1), null, new Integer(1)));
		spinner_6.setBounds(550, 62, 107, 20);
		panel.add(spinner_6);

		final JSpinner spinner_7 = new JSpinner(new SpinnerNumberModel(0.1,
				0.1, 10, 0.1));
		spinner_7.setBounds(550, 116, 107, 20);

		panel.add(spinner_7);

		final JCheckBox chckbxShowGraphgnuplot = new JCheckBox(
				"Show Graph(gnuplot)");
		chckbxShowGraphgnuplot.setBounds(455, 204, 202, 23);
		panel.add(chckbxShowGraphgnuplot);

		txtTrainDataFile = new JTextField();
		txtTrainDataFile.setText("Train data file (csv)");
		txtTrainDataFile.setBounds(12, 255, 272, 19);
		panel.add(txtTrainDataFile);
		txtTrainDataFile.setColumns(10);

		txtClassDataFile = new JTextField();
		txtClassDataFile.setText("Class data file (csv)");
		txtClassDataFile.setColumns(10);
		txtClassDataFile.setBounds(12, 308, 272, 19);
		panel.add(txtClassDataFile);

		JButton btnNewButton = new JButton("Browse");
		btnNewButton.addActionListener(new ActionListener() {

			@Override
			public void actionPerformed(ActionEvent arg0) {
				File file1 = choose();
				txtTrainDataFile.setText(file1.getAbsolutePath());
				sel1 = true;
			}
		});
		btnNewButton.setBounds(309, 252, 117, 25);
		panel.add(btnNewButton);

		JButton btnBrowse = new JButton("Browse");
		btnBrowse.addActionListener(new ActionListener() {

			@Override
			public void actionPerformed(ActionEvent e) {
				File file1 = choose();
				txtClassDataFile.setText(file1.getAbsolutePath());
				sel2 = true;
			}
		});

		btnBrowse.setBounds(309, 305, 117, 25);
		panel.add(btnBrowse);
		textArea.setBounds(22, 347, 471, 62);

		panel.add(textArea);
		rdbtnNewRadioButton.setBounds(455, 253, 218, 23);
		chckbxCascadetraining.setVisible(false);
		panel.add(rdbtnNewRadioButton);

		rdbtnNewRadioButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent arg0) {
				if (rdbtnNewRadioButton.isSelected()) {
					chckbxGpuSupport.setVisible(true);
					chckbxCascadetraining.setVisible(false);
					lblSvmCount.setText("SVM Count");
					rdbtnNewRadioButton_1.setSelected(false);
					lblCparameter.setText("C parameter");
					lblFrom.setText("From");
					lblTo.setText("To");
					lblGammaParameter.setText("Gamma parameter");
					lblStep.setText("Step");
					label.setText("Step");
					spinner.setModel(new SpinnerNumberModel(1, 1, 50, 1));
					spinner_1.setEnabled(true);
					spinner_3.setEnabled(true);
					spinner_2
							.setModel(new SpinnerNumberModel(0.1, 0.1, 10, 0.1));
					spinner_6.setModel(new SpinnerNumberModel(new Integer(1),
							new Integer(1), null, new Integer(1)));

					spinner_7
							.setModel(new SpinnerNumberModel(0.1, 0.1, 10, 0.1));
				}
			}
		});

		rdbtnNewRadioButton_1.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent arg0) {
				if (rdbtnNewRadioButton_1.isSelected()) {
					chckbxGpuSupport.setVisible(false);
					chckbxCascadetraining.setVisible(true);
					lblSvmCount.setText("ANN count");
					rdbtnNewRadioButton.setSelected(false);
					lblCparameter.setText("Number of Layers");
					lblFrom.setText("");
					lblTo.setText(null);
					lblGammaParameter.setText("Number of nuerons");
					lblStep.setText("max Epochs");
					label.setText("desired error");
					spinner.setModel(new SpinnerNumberModel(4, 3, 4, 1));
					spinner_1.setEnabled(false);
					spinner_3.setEnabled(false);
					spinner_2
							.setModel(new SpinnerNumberModel(300, 10, 10000, 1));
					spinner_6
							.setModel(new SpinnerNumberModel(300, 10, 10000, 1));
					// spinner_7.set
					spinner_7.setModel(new SpinnerNumberModel((double) 0.001,
							0.0001, 1.0, 0.001));
				}
			}
		});
		rdbtnNewRadioButton_1.setBounds(455, 306, 149, 23);

		panel.add(rdbtnNewRadioButton_1);
		chckbxCascadetraining.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				if (chckbxCascadetraining.isSelected()) {
					spinner_2.setValue(100);
				} else
					spinner_2.setValue(300);
			}
		});
		chckbxCascadetraining.setBounds(455, 159, 202, 23);
		rdbtnNewRadioButton.setSelected(true);
		panel.add(chckbxCascadetraining);
		chckbxGpuSupport.setBounds(544, 333, 129, 23);

		panel.add(chckbxGpuSupport);

		btnLaunch.addActionListener(new ActionListener() {

			@Override
			public void actionPerformed(ActionEvent e) {

				if (!sel1 | !sel2) {
					JOptionPane.showMessageDialog(null, "Select files",
							"not selected", 1);
					return;
				}
				final String param[] = new String[13];

				if (rdbtnNewRadioButton.isSelected()) {

					param[0] = spinner_1.getValue().toString();
					param[1] = spinner_4.getValue().toString();
					param[2] = spinner_3.getValue().toString();
					param[3] = spinner_6.getValue().toString();
					param[4] = spinner_7.getValue().toString();
					param[5] = spinner_5.getValue().toString();
					param[6] = txtTrainDataFile.getText();
					param[7] = txtClassDataFile.getText();
					if (rdbtnClassification.isSelected())
						param[8] = "true";
					else
						param[8] = "false";
					if (chckbxShowGraphgnuplot.isSelected())
						param[9] = "true";
					else
						param[9] = "false";
					param[10] = spinner.getValue().toString();
					param[11] = spinner_2.getValue().toString();
					if (chckbxGpuSupport.isSelected())
						param[12] = " --gpu=true";
					else
						param[12] = "";
				} else {
					if (chckbxCascadetraining.isSelected())
						param[1] = "true";
					else
						param[1] = "false";
					// param[1] = spinner_1.getValue().toString();
					param[0] = spinner_4.getValue().toString();
					param[2] = spinner.getValue().toString();
					param[3] = spinner_2.getValue().toString();
					param[4] = spinner_6.getValue().toString();
					param[5] = spinner_5.getValue().toString();
					param[6] = txtTrainDataFile.getText();
					param[7] = txtClassDataFile.getText();
					if (rdbtnClassification.isSelected())
						param[8] = "true";
					else
						param[8] = "false";
					if (chckbxShowGraphgnuplot.isSelected())
						param[9] = "true";
					else
						param[9] = "false";
					param[10] = spinner_2.getValue().toString();

					param[11] = spinner_7.getValue().toString();

				}
				// TextArea t = new TextArea();
				// new
				// JFrame("Output").getContentPane().add(t).setVisible(true);
				SwingUtilities.invokeLater(new Runnable() {
					public void run() {
						if (rdbtnNewRadioButton.isSelected())
							Processing.DoProcessing(param, textArea, true);
						else {
							Processing.DoProcessing(param, textArea, false);
						}
					}
				});

			}
		});

	}
}
