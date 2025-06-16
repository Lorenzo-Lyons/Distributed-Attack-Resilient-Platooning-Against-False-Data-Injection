import numpy as np


active_cars = np.array([False, False, True , True])
attack_detected_states = np.array([False, False, False, False])

# car_order = np.array([1,2,3,4])
# car_order = np.array([4,1,2,3])
car_order = np.array([3,4,1,2])


print('car_order = ', car_order)

# determine if some cars are inactive and remove them from teh rearranging logic
inactive_car_numbers = np.arange(1,5)[~active_cars]
inactive_car_indexes = np.where(np.isin(car_order, inactive_car_numbers))[0]
first_inactive_car_index = np.min(inactive_car_indexes)
false_indices = np.where(~active_cars)[0]

car_order_active = car_order # set to default and then will be changed if needed
car_order_inactive = []
if np.sum(~active_cars) > 0:
    if np.all(~active_cars[car_order[first_inactive_car_index:]-1]):
        # remove the last cars from the rearranging logic
        car_order_active = car_order[:first_inactive_car_index]
        car_order_inactive = car_order[first_inactive_car_index:]
    else:
        #print('some cars after the first inactive car are active')
        pass
else:
    pass


print('car_order_active = ', car_order_active)
print('car_order_inactive = ', car_order_inactive)

