# 0bdcd9d680f9009e85d94da9d68ca8e3c59a3443e78f99f5a3cca86331dd22045aec92623d0e255a501a834e23b8798fc4b96c7c686714c6d991bfbf496ec18d
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.exceptions import RuntimeInvalidStateError, RuntimeJobFailureError

# Initialize the Qiskit runtime service
service = QiskitRuntimeService(
    channel='ibm_quantum',
    instance='ibm-q/open/main',
    token='0bdcd9d680f9009e85d94da9d68ca8e3c59a3443e78f99f5a3cca86331dd22045aec92623d0e255a501a834e23b8798fc4b96c7c686714c6d991bfbf496ec18d'  # Insert your real token here
)

# List of job IDs
job_ids = [
    'cp223uu55i8ossfbs2v0', 'cp2238tp5gj47v11rg6g', 'cp222jbjf1plghcmc060', 'cs221yfkfpw00080eks0',
    'cs221k64amg00087ghtg', 'cs22195965y0008527z0', 'cs21xv7kfpw00080ekjg', 'cs212wb4amg00087ggf0',
    'cs20h0wyhpyg008ajgf0', 'cs208gjyhpyg008ajg50', 'cs203yfkfpw00080eh20', 'cs202gtbqt7g0081xsdg',
    'cs201me4amg00087gg3g', 'cs2009hkfpw00080egv0', 'cs1zznykfpw00080egtg', 'cs1zz4cyhpyg008ajftg',
    'cs1zymj4amg00087gg00', 'cs1zxvq965y00085259g', 'cs1yg3ryhpyg008ajedg', 'cs1xmc9965y0008522cg',
    'cp1tj2dp5gj47v11dulg', 'cp1tiak7eeiht4g7a1v0', 'cp1tghm55i8ossfbe74g', 'cp1tg6m0p89ot4jpaf60',
    'cp1t94e0p89ot4jp9kfg', 'cp1t3pk7eeiht4g787j0', 'cp1t3ks7eeiht4g78730', 'cp1t3gm55i8ossfbcjkg',
    'cp1t2uu0p89ot4jp8qt0', 'cp1sv53jf1plghcls0jg', 'cp1suutp5gj47v11bdt0', 'cp1sue47eeiht4g77hn0',
    'cp1sspe0p89ot4jp80dg', 'cp1ssfu0p89ot4jp7va0', 'cp1sri3jf1plghclrhd0', 'cp1sr43jf1plghclrg0g',
    'cp1sqajjf1plghclrcq0', 'cp1sp8c7eeiht4g76sug', 'cp1sp0dp5gj47v11akrg', 'cp1so59aao20lioanci0',
    'cp1sntpaao20lioanbkg', 'cp1sn7dp5gj47v11accg', 'cp1sn03jf1plghclqus0', 'cp1smnu55i8ossfbattg',
    'cp1sm7k7eeiht4g76fn0', 'cp1sm1655i8ossfbar1g', 'cp1slr655i8ossfbaq10', 'cp1slm47eeiht4g76dh0',
    'cp1slfe55i8ossfbaoeg', 'cp1sl0paao20lioan00g', 'cp1skh5p5gj47v11a0h0', 'cp1sk2lp5gj47v119ufg',
    'cp1sj3m0p89ot4jp6m30', 'cp1sit3jf1plghclqcq0', 'cp1si6dp5gj47v119lpg', 'cp1shprjf1plghclq89g',
    'cp1sfmdp5gj47v119bk0', 'cp1sf4lp5gj47v1199cg', 'cp1scvtp5gj47v118vk0', 'cp1sclu0p89ot4jp5r7g',
    'cp1sbue0p89ot4jp5ngg', 'cp1sbdm0p89ot4jp5kog', 'cp1saqm0p89ot4jp5idg', 'cp1safk7eeiht4g74t20',
    'cp1sa45p5gj47v118i5g', 'cp1s9prjf1plghclp4h0', 'cp1s9d5p5gj47v118en0', 'cp1s91k7eeiht4g74lgg',
    'cp1s85lp5gj47v1188kg', 'cp1s6rc7eeiht4g74cn0', 'cp1s54e55i8ossfb8evg', 'cp1s4kk7eeiht4g74310',
    'cp1s40e0p89ot4jp4jp0', 'cp1s3h5p5gj47v117kog', 'cp1s37bjf1plghclo8jg', 'cp1s2rbjf1plghclo71g',
    'cp1s2h5p5gj47v117gg0', 'cp1s2bjjf1plghclo50g', 'cp1s255p5gj47v117f30', 'cp1s1urjf1plghclo3c0',
    'cp1s1ktp5gj47v117d4g', 'cp1s1f9aao20lioakbvg', 'cp1s19rjf1plghclo0sg', 'cp1s1560p89ot4jp47v0',
    'cp1s0v3jf1plghclnv90', 'cp1s0ps7eeiht4g73i70', 'cp1s0ihaao20lioak79g', 'cp1s0dk7eeiht4g73gm0',
    'cp1s091aao20lioak610', 'cp1s03u0p89ot4jp42ng', 'cp1rvvm0p89ot4jp425g', 'cp1rvprjf1plghclnpsg',
    'cp1rvm5p5gj47v11742g', 'cp1rvgpaao20lioak2pg', 'cp1rvcu55i8ossfb7mkg', 'cp1rv7haao20lioak1fg',
    'cp1rv3m0p89ot4jp3uag', 'cp1rur9aao20lioajvng', 'cp1rumlp5gj47v116vk0', 'cp1ruhu55i8ossfb7itg',
    'cp1ruc5p5gj47v116u50', 'cp1rtvhaao20lioajs80', 'cp1rtpm55i8ossfb7fk0', 'cp1rtic7eeiht4g734g0',
    'cp1rtck7eeiht4g733rg', 'cp1rt6m55i8ossfb7d80', 'cp1rsvu55i8ossfb7ca0', 'cp1rsl9aao20lioajmk0',
    'cp1rsbrjf1plghclnaig', 'cp1rs7bjf1plghcln930', 'cp1rrkhaao20lioajhbg', 'cp1rr8rjf1plghcln4hg',
    'cp1rr2rjf1plghcln3ng', 'cp1rqtbjf1plghcln2u0', 'cp1rqjrjf1plghcln1c0', 'cp1rqce55i8ossfb6vt0',
    'cp1rq41aao20lioajb00', 'cp1rp3u0p89ot4jp351g', 'cp1rotu0p89ot4jp3400', 'cp1rndu0p89ot4jp2slg',
    'cp1rn5655i8ossfb6h80', 'cp1rmnlp5gj47v115q70', 'cp1rludp5gj47v115msg', 'cp1rlbbjf1plghclm9a0',
    'cp1rkh60p89ot4jp2f40', 'cp1rk7k7eeiht4g71q5g', 'cp1rjubjf1plghclm3d0', 'cp1rjfhaao20lioaice0',
    'cp1ria5p5gj47v11581g', 'cp1rhs3jf1plghcllru0', 'cp1rhem0p89ot4jp23lg', 'cp1rh5dp5gj47v1154d0',
    'cp1rgrk7eeiht4g71ek0', 'cp1rgj655i8ossfb5n50', 'cp1rg4lp5gj47v1151cg', 'cp1rfr1aao20lioahvt0',
    'cp1rfe60p89ot4jp1sl0', 'cp1re747eeiht4g7151g', 'cp1rdhu55i8ossfb5bc0', 'cp1rc5paao20lioahhkg',
    'cp1r9utp5gj47v1149og', 'cp1r8f1aao20lioah2c0', 'cp1r83bjf1plghclkl1g', 'cp1r7r9aao20lioagvv0',
    'cp1r7fe55i8ossfb4ic0', 'cp1r5phaao20lioagnk0', 'cp1r59s7eeiht4g6vvgg', 'cp1r4mdp5gj47v113i1g',
    'cp1r30u0p89ot4jp06n0', 'cp1r2mhaao20lioag4h0', 'cp1r29bjf1plghcljkc0', 'cp1r1qhaao20lioaful0',
    'cp1r0t655i8ossfb3dbg', 'cp1r0khaao20lioafog0', 'cp1r0a47eeiht4g6v0jg', 'cp1r00m55i8ossfb37eg',
    'cp1qv8k7eeiht4g6uq70', 'cp1quc1aao20lioafcgg', 'cp1qu35p5gj47v112bjg', 'cp1qsa60p89ot4jov1gg',
    'cp1qrfm55i8ossfb2fq0', 'cp1qomm55i8ossfb20bg', 'cp1qkb47eeiht4g6t130', 'cp1qim47eeiht4g6so0g',
    'cp1qifm55i8ossfb0t60', 'cp1qi3rjf1plghclgsi0', 'cp1qholp5gj47v11087g', 'cp1qhfu55i8ossfb0na0',
    'cp1qgplp5gj47v1102u0', 'cp1qgg60p89ot4josv60', 'cp1qf7haao20lioacqp0', 'cp1qd6m55i8ossfavt3g',
    'cp1qd1haao20lioaccqg', 'cp1qcrhaao20lioacbt0', 'cp1qchm0p89ot4jos790', 'cp1qcde0p89ot4jos6e0',
    'cp1qbv5p5gj47v10v6fg', 'cp1q5du55i8ossfauemg', 'cp1q4rm55i8ossfaub90', 'cp1q4h60p89ot4joqno0',
    'cp1q43bjf1plghcle7ng', 'cp1q3itp5gj47v10tktg', 'cp1q2s47eeiht4g6pm5g', 'cp1q2h655i8ossfatte0',
    'cp1pjqm55i8ossfarb1g', 'cp1pjq9aao20lioa7us0', 'cs2memj965y0008543mg'
]

def retrieve_job_data(job_id):
    try:
        job = service.job(job_id)
        result = job.result()
        return result
    except (RuntimeInvalidStateError) as e:
        print(f"Skipping job {job_id}: {str(e)}")
        return None
    except Exception as e:
        print(f"Error retrieving job {job_id}: {str(e)}")
        return None

# Retrieve and collect data
data = []
for job_id in job_ids:
    result = retrieve_job_data(job_id)
    if result:
        try:
            evs_value = result[0].data.evs
            stds_value = result[0].data.stds
            if isinstance(evs_value, (list, np.ndarray)) and len(evs_value) > 0:
                data.append((evs_value, stds_value))
        except Exception as e:
            print(f"Error processing job {job_id}: {str(e)}")

# Check if data is properly retrieved
print(f"Retrieved data from {len(data)} jobs.")

# Preprocess the data
def preprocess_data(data):
    # Ensure each element in data is a list or array
    valid_data = [evs for evs, stds in data if isinstance(evs, (list, np.ndarray)) and len(evs) > 0]
    if not valid_data:
        raise ValueError("No valid data to process.")
    
    # Find the maximum length of the evs arrays
    max_length = max(len(evs) for evs in valid_data)
    
    # Pad all evs arrays to the same length
    processed_data = np.array([np.pad(evs, (0, max_length - len(evs)), 'constant') for evs in valid_data])
    
    # Generate features (mean, std, max, min)
    features = np.array([
        [np.mean(evs), np.std(evs), np.max(evs), np.min(evs)]
        for evs in processed_data
    ])
    
    return features

if len(data) > 0:
    preprocessed_data = preprocess_data(data)

    # Create a graph from the result data
    G = nx.Graph()

    for i, (evs, stds) in enumerate(data):
        # Add nodes for evs and stds
        G.add_node(f'evs_{i}', label='evs', value=evs)
        G.add_node(f'stds_{i}', label='stds', value=stds)
        # Add edge between evs and stds
        G.add_edge(f'evs_{i}', f'stds_{i}', weight=np.mean(stds+evs))

    # Plotting the graph
    pos = nx.spring_layout(G)
    labels = nx.get_node_attributes(G, 'label')
    values = nx.get_node_attributes(G, 'value')

    nx.draw(G, pos, with_labels=True, labels=labels, node_size=1000, node_color='blue', font_size=11, edge_color='black')
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f'{d["weight"]:.2f}' for u, v, d in G.edges(data=True)})

    for node, (x, y) in pos.items():
        value = values[node]
        if isinstance(value, np.ndarray):
            text = '\n'.join([f"{v:.2e}" for v in value])
        else:
            text = f"{value:.2e}"
        plt.text(x**2, y**2, s=text, bbox=dict(facecolor='blue', alpha=0.05), horizontalalignment='center')

    plt.title('Graph Representation of Quantum Result')
    plt.show()
else:
    print("No valid data to process.")