# -*- coding: utf-8 -*-
"""
DTH SLA Simulation - Evaluate headcount, pay, and cost
by changing the cutoff time / SLA for DTH jobs.
"""

#Import relevant modules
import pandas as pd
import simpy
import numpy
import statistics
import random
import matplotlib.pyplot as plt
import math


#Create empty lists for tracking wait and idle times
# global arrival_times
# global customers
wait_times = []
idle_times = []
installs = []
customers = []
requests = []
sale_times = []
customer_iterator = []
# customers = numpy.ndarray([1, 2])
# customers[0,0] = 0
# customers[0,1] = 0
arrival_times = []
time_of_install = []
install_lengths = []
time_of_complete = []
same_day = []
next_day = []
same_next_day = []
#market1 = []


##Create times unavailble for install
simulated_hours = 168
length_choices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
probs = [0.05, .25, .3, .15, .1, .05, .025, .025, .025, .025]


# load the mean file 
df = pd.read_excel('final_mean.xlsx')


#Create and initialize the environment
class Market(object):
    def __init__(self, env, num_pros, market_size):
        """
        Iinitialize the environment with pro resource count
        """
        self.env = env
        self.pro = simpy.Resource(env, num_pros)#, monitored=True) #Create DTH pro resource
     
        
        
    def complete_install(self, env, customer, request):
        """
        Method to simulate the length of install
        """
       # install_duration = numpy.random.poisson(lam = 4) #Should replace with install duration distribution
        install_duration = random.choices(length_choices, weights=probs, k=1)[0]
        install_lengths.append(install_duration)
        
        #print(f'At {env.now}, {self.pro.count} of {self.pro.capacity} slots are allocated')
        #print(f'{self.pro.count} of {self.pro.capacity} slots are allocated.')
        # print(f'  Queued events: {self.pro.queue}')
        
        #Find opening and closing hour for current day
        request_time = env.now
        day = math.floor(request_time/24)
        open_hour = 8 + day*24
        close_hour = 20 + day*24
        if day + 1 < simulated_hours/24:
            open_hour_tomorrow = 9 + (day+1)*24
        
        #are we currently open
        if request_time < open_hour:
            wait_time = open_hour - request_time
            yield self.env.timeout(wait_time) 
        elif request_time > close_hour and day + 1 < simulated_hours/24:
            wait_time = open_hour_tomorrow - request_time
            yield self.env.timeout(wait_time) 
        elif request_time > close_hour and day + 1 > simulated_hours/24:
            wait_time = simulated_hours - request_time
            yield self.env.timeout(wait_time) 
        else:
            wait_time = 0
        
        
        install_time = env.now
        time_of_install.append(install_time)
        yield self.env.timeout(install_duration) 
        #self.pro.release(request)
        time_of_complete.append(env.now)
        installs.append(1)
        # print(f'It is {env.now} and an install was just completed')
        # print(f'{self.pro.count} of {self.pro.capacity} slots are allocated.')
        #installs_pending = len(self.pro.queue)
        #cumulative_requests = sum(requests)
        #completed_installs = sum(installs)
        #print(f'  Installs in progress or in queue: {installs_pending}')
        # print(f'  Completed Installs: {completed_installs}')
        # print(f'  Cumulative Requests: {cumulative_requests}')
        # print(f'  {self.pro.count} of {self.pro.capacity} slots are allocated')


#Methods to Move Through the Environment
def customer_sale(env,customer,market):
    """
    Method to request pro resource when customer buys a system and wants an install
    """
    arrival_time = env.now #Customer purchases a system
    #arrival_times.append(arrival_time)
    r = 0
    
    with market.pro.request() as request: #Customer makes a request for pro; with releases pro once pro completes install
        #request = market.pro.request()
        #print(f'It is {env.now} and 1 request was made')
        requests.append(1)
        day_of_request = math.floor(env.now/24)
        yield request #customer waits for pro to become available
        #requests.append(1)
        r += 1
        #print(r)
        #print(f'It is {env.now} and 1 request was made')
        #requests.append(1)
        yield env.process(market.complete_install(env, customer, request)) #Customer has available pro complete install
    day_of_install = math.floor(env.now/24)   
    if day_of_request == day_of_install:
        same_day.append(1)
        next_day.append(0)
        same_next_day.append(1)
    elif day_of_install == day_of_request+1:
        same_day.append(0)
        next_day.append(1)
        same_next_day.append(1)
    else:
        same_day.append(0)
        next_day.append(0)
        same_next_day.append(0)
        
    wait_times.append(env.now - arrival_time) #Time from purchase to install
  
  
#Directions for Running the Market Environment
 #Directions for Running the Market Environment
def run_market(env, num_pros, market_size):
    """
    Method that creates an instance of a market and generate customers until the simulation stops
    """ 
      
    market = Market(env, num_pros, market_size) #Create an instance from the class "Market"
    
    for customer in range(1): #Start the simulation with a few people in line
        #env.process(customer_sale(env,customer,market))
        # customers[0,0] = 0
        # customers[0,1] = 0
        # env.timeout(1)
        #customers.append(0)
        #arrival_times.append(0)
        #env.timeout(1)
        continue
        
        

    while True:
        if market_size == 1:
            day_of_week = math.floor(env.now/24) % 7
            hour_of_day = env.now - math.floor(env.now/24)*24
            df_verysmall = df[(df.ORIGINAL_OFFICE_NAME == 'Very Small') & (df.Day == day_of_week)].reset_index()
            x = df_verysmall.iloc[hour_of_day]['Mean']
            if x == 0:
                arrival_time = env.now #Customer purchases a system
                arrival_times.append(arrival_time)
                customers.append(0)
                yield env.timeout(1) #No customers generated in this hour
            else: 
                mean = x
                arrival_time = env.now
                hourly_new_customer = numpy.random.poisson(lam = mean)
                customers.append(hourly_new_customer)
                customer += hourly_new_customer
                arrival_times.append(arrival_time)
                i=1
                #print(f'It is {env.now} and {hourly_new_customer} sale(s) came in')
                while i <= hourly_new_customer:
                    sale_times.append(env.now)
                    customer_iterator.append(1)
                    env.process(customer_sale(env,customer,market))
                    i+=1
                yield env.timeout(1) #This hour passes and customers are generated below
                 
                
        elif market_size == 2:
            day_of_week = math.floor(env.now/24) % 7
            hour_of_day = env.now - math.floor(env.now/24)*24
            df_verysmall = df[(df.ORIGINAL_OFFICE_NAME == 'Small') & (df.Day == day_of_week)].reset_index()
            x = df_verysmall.iloc[hour_of_day]['Mean']
            if x == 0:
                arrival_time = env.now #Customer purchases a system
                arrival_times.append(arrival_time)
                customers.append(0)
                yield env.timeout(1) #No customers generated in this hour
            else: 
                mean = x
                arrival_time = env.now
                hourly_new_customer = numpy.random.poisson(lam = mean)
                customers.append(hourly_new_customer)
                customer += hourly_new_customer
                arrival_times.append(arrival_time)
                i=1
                #print(f'It is {env.now} and {hourly_new_customer} sale(s) came in')
                while i <= hourly_new_customer:
                    sale_times.append(env.now)
                    customer_iterator.append(1)
                    env.process(customer_sale(env,customer,market))
                    i+=1
                yield env.timeout(1) #This hour passes and customers are generated below
                
                
        elif market_size == 3:
            day_of_week = math.floor(env.now/24) % 7
            hour_of_day = env.now - math.floor(env.now/24)*24
            df_verysmall = df[(df.ORIGINAL_OFFICE_NAME == 'Medium') & (df.Day == day_of_week)].reset_index()
            x = df_verysmall.iloc[hour_of_day]['Mean']
            if x == 0:
                arrival_time = env.now #Customer purchases a system
                arrival_times.append(arrival_time)
                customers.append(0)
                yield env.timeout(1) #No customers generated in this hour
            else: 
                mean = x
                arrival_time = env.now
                hourly_new_customer = numpy.random.poisson(lam = mean)
                customers.append(hourly_new_customer)
                customer += hourly_new_customer
                arrival_times.append(arrival_time)
                i=1
                #print(f'It is {env.now} and {hourly_new_customer} sale(s) came in')
                while i <= hourly_new_customer:
                    sale_times.append(env.now)
                    customer_iterator.append(1)
                    env.process(customer_sale(env,customer,market))
                    i+=1
                yield env.timeout(1) #This hour passes and customers are generated below
                
                
        elif market_size == 4:
            day_of_week = math.floor(env.now/24) % 7
            hour_of_day = env.now - math.floor(env.now/24)*24
            df_verysmall = df[(df.ORIGINAL_OFFICE_NAME == 'Large') & (df.Day == day_of_week)].reset_index()
            x = df_verysmall.iloc[hour_of_day]['Mean']
            if x == 0:
                arrival_time = env.now #Customer purchases a system
                arrival_times.append(arrival_time)
                customers.append(0)
                yield env.timeout(1) #No customers generated in this hour
            else: 
                mean = x
                arrival_time = env.now
                hourly_new_customer = numpy.random.poisson(lam = mean)
                customers.append(hourly_new_customer)
                customer += hourly_new_customer
                arrival_times.append(arrival_time)
                i=1
                #print(f'It is {env.now} and {hourly_new_customer} sale(s) came in')
                while i <= hourly_new_customer:
                    sale_times.append(env.now)
                    customer_iterator.append(1)
                    env.process(customer_sale(env,customer,market))
                    i+=1
                yield env.timeout(1) #This hour passes and customers are generated below
                
        elif market_size == 5:
            day_of_week = math.floor(env.now/24) % 7
            hour_of_day = env.now - math.floor(env.now/24)*24
            df_verysmall = df[(df.ORIGINAL_OFFICE_NAME == 'Very Large') & (df.Day == day_of_week)].reset_index()
            x = df_verysmall.iloc[hour_of_day]['Mean']
            if x == 0:
                arrival_time = env.now #Customer purchases a system
                arrival_times.append(arrival_time)
                customers.append(0)
                yield env.timeout(1) #No customers generated in this hour
            else: 
                mean = x
                arrival_time = env.now
                hourly_new_customer = numpy.random.poisson(lam = mean)
                # cust_to_append = numpy.ndarray([1, 2])
                # cust_to_append[0,0] = 1
                # cust_to_append[0,1] = 3
                # customers = numpy.r_[customers, cust_to_append]
                customers.append(hourly_new_customer)
                customer += hourly_new_customer
                arrival_times.append(arrival_time)
                i=1
                #print(f'It is {env.now} and {hourly_new_customer} sale(s) came in')
                while i <= hourly_new_customer:
                     #Customer purchases a system
                    sale_times.append(env.now)
                    customer_iterator.append(1)
                    #customers += 1
                    env.process(customer_sale(env,customer,market))
                    #requests.append(1)
                    i+=1
                    
                yield env.timeout(1) #This hour passes and customers are generated below
                
                
        else:
           print('Enter a valid market')
           break
            
        
#Outputs to Track
def get_average_wait_time(wait_times):
    average_wait = statistics.mean(wait_times)
    hours, frac_hours = divmod(average_wait, 1)
    minutes = frac_hours * 60
    return round(hours), round(minutes)


#Get User Input
def get_user_input():
    num_pro_input = input("Input # of DTH pros working: ")
    num_pros = int(num_pro_input)
    return num_pros

#Get User Input
def get_user_input_market():
    market_input = input("Input the market size (1-s/2-vs/3-m/4-l/5-vl): ")
    market_size = int(market_input)
    return market_size

#Get User Input
def number_of_trials():
    trial_input = input("Input the number of trials: ")
    trial = int(trial_input)
    return trial


    

#Finalize Setup
def main():
    #Setup
    #Prompt user for input
    num_pros = get_user_input() 
    market_size = get_user_input_market()
    trial = number_of_trials()
    
    for i in range(1,trial+1,1):
        
        random.seed(i) #Set seed for random
        numpy.random.seed(i) #Set seed for numpy
        
        #Run the simulation
        env = simpy.Environment() #Create environment and assign to variable env
        env.process(run_market(env,num_pros, market_size)) #Call process and run_market
        env.run(until=simulated_hours) #Length of simulation
    
    
        
     
    
    ## Export ungrouped data   
    #Customer_simulated_ungrouped = pd.DataFrame(list(zip(customers, list(range(0,144,1)))), columns =['Customer_Count', 'time'])    
    Customer_simulated_ungrouped = pd.DataFrame(list(zip(arrival_times, customers)), columns =['time', 'Customer_Count'])    
    Customer_simulated_ungrouped.to_csv('CustomerArrivalUngrouped.csv') 
    
    ##Fill hours where no customer arrives to 0
    # arrival_times_all = pd.DataFrame(list(range(0,144,1)))
    # arrival_times_all.columns = ['time']
    # arrival_times_fin = arrival_times_all.merge(Customer_simulated_ungrouped, how = 'left', on = 'time')
    # arrival_times_fin['Customer_Count'] = arrival_times_fin['Customer_Count'].fillna(0)
    # arrival_times_fin = arrival_times_fin.groupby('time')['Customer_Count'].mean()
    # arrival_times_fin = pd.DataFrame(arrival_times_fin)
    
    ##Group results into hours by average
    Customer_simulated = Customer_simulated_ungrouped.groupby('time')['Customer_Count'].mean()
    #Customer_simulated = Customer_simulated_ungrouped.groupby('time')['Customer_Count'].sum()
    #Customer_simulated = pd.DataFrame(customers)
    #Customer_simulated.columns = ['time', 'Customer_Count']
    Customer_simulated.to_csv('CustomerArrival.csv') 
    
   
    #Customer_simulated_count = Customer_simulated['Customer_Count'].sum()
    Customer_simulated_count = sum(Customer_simulated)
    
    #View results
    mins, secs = get_average_wait_time(wait_times) 
    #customer_count = round(data['Customer'].sum())
    customer_count = round(Customer_simulated_count)
    install_count = round(len(installs)/trial)
    same_day_df = pd.DataFrame(same_day)
    same_day_df.columns = ['Same_day']
    same_day_percent = round(sum(same_day_df.Same_day)/len(same_day_df)*100, 2)
    next_day_df = pd.DataFrame(next_day)
    next_day_df.columns = ['Next_day']
    next_day_percent = round(sum(next_day_df.Next_day)/len(next_day_df)*100, 2)
    same_next_day_df = pd.DataFrame(same_next_day)
    same_next_day_df.columns = ['Same_Next_day']
    same_next_day_percent = round(sum(same_next_day_df.Same_Next_day)/len(same_next_day_df)*100, 2)
    #request_count = sum(requests)/trial
    print(
        "Running simulation...",
        f"\nThe average wait time is {mins} hours and {secs} minutes." ,
       ) 
   
    print(f"Vivint sold {customer_count} accounts.")
    #print(f"{request_count} requests were made to the pro resource.")
    print(f"Vivint installed {install_count} customers.")
    print(f"Same Day Percent {same_day_percent} %.")
    print(f"Next Day Percent {next_day_percent} %.")
    print(f"Same/Next Day Percent {same_next_day_percent} %.")
    # print(f'TimeAverage no. waiting:',self.pro.waitMon.timeAverage())
    # print(f'(Number) Average no. waiting:',server.waitMon.mean())
    # print(f'(Number) Var of no. waiting:',server.waitMon.var())
    # print(f'(Number) SD of no. waiting:',sqrt(server.waitMon.var()))
    # print(f'TimeAverage no. in service:',server.actMon.timeAverage())
    # print(f'(Number) Average no. in service:',server.actMon.mean())
    # print(f'(Number) Var of no. in service:',server.actMon.var())
    # print(f'(Number) SD of no. in service:',sqrt(server.actMon.var()))
    
    # Histogram of Wait Times 
    wait_hist = plt.hist(wait_times, bins = int(simulated_hours/24))
    plt.title('Histogram: Wait Times')
    plt.show(wait_hist)
    
        
    # Actual Arrival time - NOT SCALABLE - Bring it outside of the Def Market()   
    if market_size == 1:
        df_mar = df[(df.ORIGINAL_OFFICE_NAME == 'Very Small')].reset_index()
    elif market_size == 2:
        df_mar = df[(df.ORIGINAL_OFFICE_NAME == 'Small')].reset_index()
    elif market_size == 3:
        df_mar = df[(df.ORIGINAL_OFFICE_NAME == 'Medium')].reset_index()
    elif market_size == 4:
        df_mar = df[(df.ORIGINAL_OFFICE_NAME == 'Large')].reset_index()
    elif market_size == 5:
        df_mar = df[(df.ORIGINAL_OFFICE_NAME == 'Very Large')].reset_index()
 
    
    # Start time of install 
    time_of_install_df = pd.DataFrame(time_of_install, columns = ['time'])
    time_of_install_df['number'] = 1
    time_of_install_df = time_of_install_df.groupby('time')['number'].count()
   
    
    #Histogram of Install Times
    install_length_hist = plt.hist(install_lengths, bins = 30) 
    plt.title('Histogram: Install Lengths')
    plt.show(install_length_hist)

    
    
    #print(arrival_times_df)
    #print(arrival_times)
    #print(arrival_times_fin)
    
    # Graph of Arrival and Start time of install
    fig, axes= plt.subplots(nrows=2, ncols=1,figsize=(10,12))
    axes[0].plot(Customer_simulated,label='Simulated Arrival')
    axes[0].plot(df_mar.Mean,label='Actual Arrival')
    axes[0].legend()
    axes[0].set_title('Arrival Times')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Number of Customers')
    
    axes[1].plot(time_of_install_df,label='Simulated Start time of install')
    #axes[1].plot(nopromean,label='No Promotion')
    axes[1].legend()
    axes[1].set_title('Simulated Start time of Install')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Start Time of Install')
    plt.show()
    
    time_of_install_hist = plt.hist(time_of_install, bins = 30) #Histogram of time of Install
    plt.title('Histogram: Time of Install')
    plt.show(time_of_install_hist)

    
    
  
#Run the Simulation
if __name__ == '__main__':
    main()



#Parameters
#1. Market environment - Completed 
#2. Customer Arrival - In Progress: Need Distribution + Vary by Day of Week - Arvind
#3. Install Time - In Progress: Need Distribution - Ken
#4. Cutoff time - Not Started: Need Logic - Taylor
#5. Wait Time < 48 hours
#6. Number of Pros - Completed

#Ouputs to Track
#1. Total Installs/Week
#2. Installs by Time of Day
#3. Avg. Installs / Pro / Day
#4. Avg Idle Time Blocks


# Changes Needed 
#wait_times = []
#idle_times = NOT BEING USED ANYWHERE - Check and remove this 
#customers -> Arrival time in main()

# 12142020
# Output (Install_count)
# Graph - Simulated Start Time of Install 
# Other Graphs 
