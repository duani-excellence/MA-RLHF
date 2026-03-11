export HF_ENDPOINT=https://hf-mirror.com


echo "----------------------------------------------------------------------"
echo "run examples: "
echo "./run_generation_examples.sh ./output/sft_full 512"
echo "----------------------------------------------------------------------"

model_path=$1
max_new_tokens=$2

echo "***********************************************************************"
echo "         [model] : "${model_path}
echo "[max_new_tokens] : "${max_new_tokens}
echo "***********************************************************************"



######################################### test math ################################################


PROMPT_SET=(
	# GSM8K
	'Tom has 12 apples. He gives 3 apples to each of his 4 friends. After that, he buys 10 more apples. How many apples does Tom have now?'

	'Sarah has 5 packs of pencils. Each pack contains 8 pencils. She gives 12 pencils to her classmates. How many pencils does Sarah have left?'

	# MATH
	'Let $a,$ $b,$ $c,$ $d$ be positive real numbers. Find the minimum value of \[(a + b + c + d) \left( \frac{1}{a} + \frac{1}{b} + \frac{1}{c} + \frac{1}{d} \right).\]' # 16

	'Let $f(x) = x^3 + 3x^2 + 1.$ There exist real numbers $a \neq 0$ and $b,$ such that \[f(x) - f(a) = (x - a)^2 (x - b).\]Enter the ordered pair $(a,b).$'  #boxed{(-2,1)}

	'Let $m \circ n = \frac{m+n}{mn+4}$. Compute $((\cdots((2005 \circ 2004) \circ 2003) \circ \cdots \circ 1) \circ 0)$.' #boxed{1/12)}

	'In the coordinate plane, the graph of \[|x + y - 1| + \Big| |x| - x \Big| + \Big| |x - 1| + x - 1 \Big| = 0\]is a certain curve. Find the length of this curve.' # boxed{\sqrt(2)}

	'The point $P = (-1,0)$ lies on the ellipse $4x^2 + y^2 = 4.$ Let $Q$ be another point on this ellipse, and let $d$ be the maximum length of $\overline{PQ}.$ Find $d^2.$' #\\boxed{\\frac{16}{3}}

	'If $f(x) = \frac{4x+1}{3}$ what is the value of $\left[f^{-1}(1)\right]^{-1}$?' #box{2}

)


for PROMPT in "${PROMPT_SET[@]}"; do
	echo '-------------------------- test math prompt --------------------------'
	echo ${PROMPT}
	python ./ma-rlhf/generate.py \
 	--model_name=${model_path} \
 	--prompt="${PROMPT}" \
 	--max_new_tokens=${max_new_tokens} \
	--step_generate=True
done
