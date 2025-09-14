# General
This CommunicationNetworkGenerator uses pandapower and simbench-based powergrids and applies different strategies 
to generate communication networks. The basic idea considers HVLV and MVLV transformers as routers which are 
conntected to each other has a backhaul network in a meshed-topology in a dedicated infrastructure setting - 
or in a star-like topology in a hybrid private-publicly owned infrastructure.
LV compoents are added in a star-topology to a router resembling a wireless communication infrastructure as Access Network
Bitrates can be freely choosen for Access and Backhaul Network.


# Install
Use the provided requirements.txt as newer pandapower and simbench versions lack the component naming

# how to use
1) use the given cigre_MV_LV_Graph or generate your own with the tool from https://gitlab.offis.de/fOest/modeling-and-visualizatoin-of-communication-networks.git
2) If you regenerated a new model, copy and paste it to this directory
3) you may execute the main-function in PPGraphConfig and play with the parameters