EXEC	= map
OFILES	= mcar2D.o mcar3D.o map2Dto3D.o

CC	= g++ -std=c++11

LIBS	=	-lm 

$(EXEC)	: $(OFILES)
	echo "linking $<"
	$(CC) -O2 -o $(EXEC) $(OFILES)  $(LIBS) $(INCLUDE)
	echo -n "make finished at "; date

mcar2D.o : mcar2D.cpp mcar2D.h
	$(CC) -c mcar2D.cpp

mcar3D.o : mcar3D.cpp mcar3D.h
	$(CC) -c mcar3D.cpp

map2Dto3D.o : map2Dto3D.cpp map2Dto3D.h
	$(CC) -c map2Dto3D.cpp

# Clean up after making.
clean:
	echo "removing" *.o
	$(RM) *.o
	if [ -d /tmp/$(USER)tmp ]; \
	 then echo "clearing /tmp/$(USER)tmp"; $(MAKE) tclean; fi


# Time your make.
time:
	/usr/bin/time $(MAKE)




