export HF_ENDPOINT=https://hf-mirror.com


echo "----------------------------------------------------------------------"
echo "run examples: "
echo "./run_generation_examples.sh ./output/sft_full ./output/prm_lora 512"
echo "----------------------------------------------------------------------"

model_path=$1
lora_path=$2
max_new_tokens=$3

echo "***********************************************************************"
echo "         [model] : "${model_path}
echo "          [lora] : "${lora_path}
echo "[max_new_tokens] : "${max_new_tokens}
echo "***********************************************************************"


PROMPT_SET=(
	# GSM8K
	'Tom has 12 apples. He gives 3 apples to each of his 4 friends. After that, he buys 10 more apples. How many apples does Tom have now?'

	'Sarah has 5 packs of pencils. Each pack contains 8 pencils. She gives 12 pencils to her classmates. How many pencils does Sarah have left?'

	# MATH
	'Let $a,$ $b,$ $c,$ $d$ be positive real numbers. Find the minimum value of \[(a + b + c + d) \left( \frac{1}{a} + \frac{1}{b} + \frac{1}{c} + \frac{1}{d} \right).\]'

	'how many positive two-digit integers leave a remainder of 2 when divided by 8?'

	'The solutions of the equation $z^4+4z^3i-6z^2-4zi-i=0$ are the vertices of a convex polygon in the complex plane. The area of this polygon can be expressed in the form $p^{a/b},$ where $a,$ $b,$ $p$ are positive integers, $p$ is prime, and $a$ and $b$ are relatively prime. Find $a + b + p.$'

	# omni-math
	'Find all the pairs of prime numbers $ (p,q)$ such that $ pq|5^p\plus{}5^q.$'

	'Let $\mathbb{N}$ denote the set of positive integers. Find all functions $f: \mathbb{N} \to \mathbb{N}$ such that \[ f(m+n)f(m-n) = f(m^2) \] for $m,n \in \mathbb{N}$.'

	'$x$, $y$ and $z$ are positive reals such that $x+y+z=xyz$. Find the minimum value of: \[ x^7(yz-1)+y^7(zx-1)+z^7(xy-1) \]'
)

for PROMPT in "${PROMPT_SET[@]}"; do
	echo '-------------------------- test math prompt --------------------------'
	echo ${PROMPT}
	python ./ma-rlhf/generate_step_prm.py \
 	--model_name=${model_path} \
	--lora_path=${lora_path} \
 	--prompt="${PROMPT}" \
 	--max_new_tokens=${max_new_tokens} \
	--step_generate=True
done
