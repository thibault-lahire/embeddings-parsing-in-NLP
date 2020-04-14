# Algorithms for speech and natural language processing TP2

Development of a basic probabilistic parser for French based on the CYK algorithm and the PCFG model and that is robust to unknown words.

We used nltk, PYEVALB and sklearn. Make sure you have these requirements already installed.

## Parsing

To see what the system is able to do, use the simple_test.txt file (you can modify it) and execute the command:

```
sh run.sh --parse simple_test.txt
```

When the files 'evaluation_data.parser_output' and 'evaluation_data.ground_truth' are available, you can evaluate the results of the system on the test corpus with the command:

```
sh run.sh --evaluate
```

To perform inference on the test corpus (It will replace the current files 'evaluation_data.parser_output' and 'evaluation_data.ground_truth'), use the command:

```
sh run.sh --do_inference
```