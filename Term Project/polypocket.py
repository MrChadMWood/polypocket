import numpy as np
import geopandas as gpd
import random
import osmnx as ox
from shapely.geometry import Point
from shapely.geometry import MultiPoint
from shapely.geometry import LineString
from shapely.geometry import mapping
import numpy as np
import geopandas as gpd


class StreetMap():
    # Fallback speedlimits per road type (in place of missing data)
    hwy_speeds = {
        'motorway': 70,
        'trunk': 65,
        'primary': 55,
        'secondary': 45,
        'tertiary': 35,
        'unclassified': 30,
        'residential': 25}
    
    fallback = 45 # Last resort

    
    def __init__(self, center, dist, crs='epsg:4326', hwy_speeds=hwy_speeds, fallback=fallback):
        '''
        This class was designed to provide a way for users to query for streetmap data via the 
        OpenStreetMap API, using the osmnx wrapper. It enables users to specify default values 
        for speedlimit based on road type, as speed limit data is largely missing from OSM. Alternatively, 
        default values are also provided.
        
        params: 
                center:        Tuple(Lon, Lat) coordinate points to act as centroid of street data
                dist:          Numeric distance (meters) from center to query street data for
                crs:           Defualt: 'epsg:4326'; Used for projection
                hwy_speeds:    Dict object containing {osm_road_type : fallback_speed_limit} 
                               for use in place of missing values
                fallback:      Master fallback that is used in place of anything unspecified for a missing value
                
        attributes: 
                centroid:        Returns Point object of provided center
                centroid_gdf:    Returns a projected GeoDataFrame of centroid
                gdf:             Returns the buffered polygon in a GeoDataFrame
                poly:            Returns the buffered polygon
                bounds:          Returns (minx, miny, maxx, maxy) bounds of poly
                G:               Returns Graph object containing street data within poly
                nodes:           Returns G nodes in GeoDataFrame
                edges:           Returns G edges in GeoDataFrame
        '''
        # Defines attributes for the center point
        self.centroid   = Point(center)
        self.centroid_gdf = gpd.GeoDataFrame([self.centroid], columns=['geometry'], crs=crs)
        
        # Defines the buffered/projected shape
        self.gdf = self.centroid_gdf.to_crs('+proj=aeqd +units=m  +x_0=0 +y_0=0')
        self.gdf = self.gdf['geometry'].buffer(dist).to_crs(crs)
        self.poly = self.gdf[0]
        
        # Defines attributes for the buffered shape boundaries
        self.bounds = self.gdf.bounds.values[0]
        
        # Queries for graph, travel speeds, bearing
        self.G = ox.graph_from_polygon(self.poly, network_type='drive_service')
        self.G = ox.speed.add_edge_speeds(self.G, hwy_speeds=hwy_speeds, fallback=fallback)
        self.G = ox.bearing.add_edge_bearings(self.G)
        
        # Converts graph to GeoDataFrames
        self.nodes, self.edges = ox.graph_to_gdfs(self.G, nodes=True, edges=True)
        

class Enviroment():
    
    def __init__(self, G, objects):
        '''
        This class was designed to provide an agent with an enviroment to train on. It also 
        provides a method to return the action space of a given node.
        
        params: 
                G:                  MultiDiGraph object generated via polypocket.StreetMap.
                objects:            Dict object containing indexs of OSMIDs that represent a house or gasstation
                
        attributes: 
                indexer:            Returns dict where keys=osmids and values=labels (label encoding)
                houses:             Returns Int64Index of s hape (n, ); [house_osmid]
                gasstations:        Returns Int64Index of shape (n, ); [gasstation_osmid]
                norm_houses:        Returns array of shape (n, ); [house_label]
                norm_gasstations:   Returns array of shape (n, ); [gasstation_label]
                nodes:              Returns array of shape (n, 2); [node-osmid, dict(attributes)]
                edges:              Returns GeoDataFrame of edges multiindexed by osmid, includes attributes length and geometry
                edges_arr:          Returns array of shape (n, 3); [edgeStart_osmid, edgeEnd_osmid, length]
                norm_nodes:         Returns array of shape (n, ); [node_label]
                norm_edges:         Returns array of shape (n, 3); [edgeStart_label, edgeEnd_label, length]
                flagged_edges:      Returns array of shape (n, 5); [(edges_arr+), bool_is_house*1, bool_is_gasstation*1]
                norm_flagged_edges: Returns array of shape (n, 5); [(norm_edges+), bool_is_house*1, bool_is_gasstation*1]
        
        methods: 
                get_actions(self, node_id): Returns array of shape (n, 2); [neighbor_node_id, length]
        '''
        # Maps each node to an index based on order of appearance
        def indexer(nodes):
            nodes_index = list(range(len(nodes)))
            return dict(zip(nodes, nodes_index))
        

        # Filters edges for only necessary attributes, returns GeoDataFrame
        def filter_edges(edges):
            attribute_dict = dict()
            for node, connection, data_dict in edges:
                edge = (node, connection)
                attribute_dict.update({edge:data_dict})  
                
            edges = gpd.GeoDataFrame(attribute_dict).T
            edges['length'] = edges['length'].astype(float)
            
            return edges[['length', 'geometry']]
        

        # Converts edges to array
        def edges_to_array(self):
            edges_arr = np.array([i for i in self.edges.index])
            attributes = self.edges['length'].to_numpy().reshape(-1,1)
            return np.append(edges_arr, attributes, axis=1)

        
        # Normalizes node IDs
        def normalize_nodes(self):
            return np.arange(len(self.nodes)).reshape(-1,1)
        
        
        # Normalized edge IDs and attributes
        def normalize_edges(self):
            df = self.edges.rename(index=self.indexer)
            df = df.drop(columns=['geometry']) 
            return np.append(np.array([*df.index]), df.values, axis=1)
        
        
        def edge_flags(self):
            # Mask of edges that travel to a target or gasstation
            edges_to_targets     = np.isin(self.norm_edges[:,1], self.norm_houses).reshape(-1,1)*1
            edges_to_gasstations = np.isin(self.norm_edges[:,1], self.norm_gasstations).reshape(-1,1)*1
            
            # Appends for use as flags to better generalize the data
            edge_flags = np.append(edges_to_targets, edges_to_gasstations, axis=1)

            return edge_flags
        
        
        # Creates dict maping a standard index to each node
        self.indexer = indexer(G.nodes)
        
        # Enviroment objects
        self.houses      = objects.get('houses')
        self.gasstations = objects.get('gasstations')
        self.norm_houses      = np.array(list(map(self.indexer.get, self.houses)))
        self.norm_gasstations = np.array(list(map(self.indexer.get, self.gasstations)))
        
        # Nodes and edges
        self.nodes = np.array(G.nodes(data=True))
        self.edges = filter_edges(G.edges(data=True))
        self.edges_arr = edges_to_array(self)
        
        # Normalized node and edge IDs
        self.norm_nodes = normalize_nodes(self)
        self.norm_edges = normalize_edges(self)
        
        # Flags what edges travel to a target or gasstation using a mask column
        self.flagged_edges      = np.append(self.edges_arr, edge_flags(self), axis=1)
        self.norm_flagged_edges = np.append(self.norm_edges, edge_flags(self), axis=1)
    
    
    # Returns available edges from a node
    def get_actions(self, node_id):   
        # Checks normalized edges first
        if node_id in self.norm_edges[:,0]:
            return self.norm_edges[self.norm_edges[:,0] == node_id][:,1:]
        # Checks default edges
        elif node_id in self.edges.index:
            return self.edges.loc[node_id]
        
        else:
            raise KeyError(f'node_id "{node_id}" was not found.')


class State():
       
    def __init__(self, enviroment, start_position, 
                 gas_level=20, kpg=18, refuel_at=0.25, gas_price=5, 
                 target_value=100):
        '''
        This class was designed to provide an agent the capability to travel 
        between edges on a graph object. The class makes a game out of the 
        task by providing options as an action space and keeping track of 
        current state information.
        
        params: 
                enviroment:        Generated via polypocket.Enviroment
                start_position:    Node_Id, should be a node available in enviroment
                gas_level:         Default:20, Gallons of gas available to agent
                kpg:               Default:18, Kilometers per gallon
                refuel_at:         Default:0.25, Threshhold percent of gas_level before agent should refuel
                gas_price:         Default:5, Cost per gallon of gas in dollars
                target_value:      Default:100, Reward for visiting a target node
        
        attributes:
                start:                Returns start_position
                distance:             Returns total distance traveled
                position:             Returns current position
                gas:                  Returns current gas_level in gallons
                max_gas:              Returns maximum gas_level in gallons
                kpg:                  Returns fuel efficiency (constant)
                refuel_at:            Returns threshhold percent of max_gas when agent will refuel
                gas_price:            Returns the price of gas per gallon (constant)
                cost:                 Returns the current total cost of the route
                route:                Returns array of shape(s,) for each state; [position|state(by index)]
                targets:              Returns array of shape(n,); [target]
                gasstations:          Returns array of shape(n,); [gasstation]
                remaining_targets:    Returns array of shape(n,); [remaining_target]
                target_value:         Returns the reward for visiting a target
                env:                  Returns array of shape(n, 5); 
                                          [edgeStart, edgeEnd, length, bool_is_remaining_target*1, bool_is_gasstation*1]
                choices:              Returns action space as array of shape(n, 4); [choice_node, edge_attribudes]
                
        methods: 
                gas_percent(self):               Returns current level of gas as percent of maximum
                choices(self):                   Returns available choices as rows in an array
                set_position(self, position):    Functional-Changes current position of agent to provided position
                                                 if position is valid choice
        '''
        # Validates start position
        if start_position not in np.array([enviroment.nodes[:,0], enviroment.norm_nodes[:,0]]):
            raise ValueError('start_position not within map.')
        
        self.start     = start_position
        self.distance  = 0 # Total distance travelled
        
        self.position  = start_position
        
        self.gas       = gas_level
        self.max_gas   = gas_level
        self.kpg       = kpg
        self.refuel_at = refuel_at
        
        self.gas_price = gas_price
        self.cost      = 0
        
        self.route       = np.array(start_position)    
        self.targets     = enviroment.norm_houses
        self.gasstations = enviroment.norm_gasstations
        
        self.remaining_targets = enviroment.norm_houses
        self.target_value = target_value
        self.env = enviroment.norm_flagged_edges
        
        self.choices = self.env[self.env[:,0]==self.position][:,1:]
        
        
    def gas_percent(self):
        return self.gas/self.max_gas
    
    
    def set_position(self, position):
        
        # Updates targets and env if position is target
        def update_targets(self):
            mask = np.where(self.remaining_targets == position) # Mask for updating targets
            self.remaining_targets = np.delete(self.remaining_targets, mask) # Updates targets
            self.env[:,3] = np.isin(self.env[:,1], self.remaining_targets)*1 # Updates enviroment
        
        # Refuels if position is gasstation
        def refuel(self):
            if self.gas_percent() <= self.refuel_at: # if gas is <= refuel at percent
                gas_needed = self.max_gas-self.gas # Defines amount of gas needed
                self.cost += gas_needed/self.gas_price # Adds cost of gas needed to route cost
                self.gas   = self.max_gas # Resets gas to mac_gas
   

        # Prevents travelling to current location
        if position == self.position:
            raise AttributeError(f'{position} already occupies the current state.')
        
        # Updates attributes if arg was valid
        if position in self.choices[:,0]:
            length         = self.choices[self.choices[:,0] == position][:,-1]
            self.distance += length
            self.gas      -= length/self.kpg
            self.position  = position
            self.choices   = self.env[self.env[:,0]==self.position][:,1:]
            self.route     = np.append(self.route, position)
        else:
            raise AttributeError(f'Can not travel to {position} from {self.position}. Refer to self.choices.')
        
        # Checks if position is target or gasstation
        if position in self.remaining_targets:
            update_targets(self)
        if position in self.gasstations:
            refuel(self)
            

def generate(streetmap, houses=None, gasstations=None):
    '''
    This function was designed specifically to be used alongside class StreetMap, 
    though it does not need to be. The function takes in a graph and two kwargs 
    with int values, instructing the function to generate n random points for houses
    or select n random nodes for gasstations. A nearest neighbor search is performed on
    each house, and the resulting neighbors will represent the agent's target 
    locations.
    
    params:
        streetmap:      Graph object generated via class StreetMap
        houses:         Int value kwarg, how many random points to generate for houses
        gasstations:    Int value kwarg, how many nodes to select randomly from streetmap
        
    returns:
        objects:    Dict object with the following key, value pairs:
                    dict(
                        points      = MultiPoint object of n_houses random points, 
                        houses      = Array of Point objects for each neighbor, 
                        connections = Array of LineStrings connecting points to houses, 
                        gasstations = Array of Point objects for each gasstation, 
                        osmids      = dict(houses=Int64Index(houses_osmids), 
                                           gasstations=Int64Index(gasstations_osmids))
                    )
                    
        note that only objects.get('osmids') is needed by other classes. The other key, value pairs
        are present for visualizations only.
    '''
    # Collection of all objects
    objects = dict({
        'points':None,
        'houses':None,
        'connections':None,
        'gasstations':None,
        'osmids':dict()}) 
    
    # Collection of object IDs
    _osmids = dict()
    
    # Exterior bounds of streetmap
    minx, miny, maxx, maxy = streetmap.bounds
    
    
    # Generates random points within bounds of polygon
    def random_points(n):
        points = []
        while len(points) < n:  
            # generates random point within bounds max(x,y) to min(x,y)
            random_point = Point([random.uniform(minx, maxx), 
                                  random.uniform(miny, maxy)])

            # Verifies points within polygon and appends
            if (random_point.within(streetmap.gdf[0])):
                points.append(random_point)
        
        return points
    
    
    # Generates house points within polygon
    def _houses(n):
        # Creates seperate collections for coordinates
        points     = MultiPoint(random_points(n))
        coords     = np.array(mapping(points).get('coordinates'))
        longitudes = coords[:,0]
        latitudes  = coords[:,1]
        
        # Queries nearest node in polygon for each rand point
        destinations = ox.distance.nearest_nodes(streetmap.G, longitudes, latitudes) 
        houses       = streetmap.nodes.loc[destinations]['geometry']
        
        # Linestrings connecting random points to their associated house
        connections = list(LineString([point, house]) for point, house in zip(points.geoms, houses))
        
        # Updates objects variables
        objects.update({
            'points':points, 
            'houses':houses.to_numpy(), 
            'connections':connections})
        _osmids.update({
            'houses':houses.index})

    
    # Generates array of random nodes taken from street network
    def _gasstations(n):
        gasstations = streetmap.nodes.sample(n=n)['geometry'] # Samples dataframe
        
        # Updates objects variables
        objects.update({
            'gasstations':gasstations.to_numpy()})
        _osmids.update({
            'gasstations':gasstations.index}) 
    
    
    # Generates args as provided
    _houses(houses)
    _gasstations(gasstations)
    
    # Updates objects variables
    objects.update({
        'osmids':_osmids,})
    
    return objects