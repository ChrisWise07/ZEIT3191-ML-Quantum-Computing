from qiskit import IBMQ

from api_keys import ibm_quantum_api_key

# save api key locally
IBMQ.save_account(ibm_quantum_api_key)
