# Detect OS
ifeq ($(OS),Windows_NT)
    RM = cmd /C del /F /Q
    EXEC_EXT = .exe
else
    RM = rm -f
    EXEC_EXT =
endif

# Compiler settings
CXX = g++
CXXFLAGS = -Wall -g -IC:/Libraries/ -std=c++17

# Define the executable and object files
EXEC = app$(EXEC_EXT)
OBJS = app.o datagenerator.o logisticregression.o

# The first rule is the default rule
# Build the executable by linking the object files
$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(EXEC) $(OBJS)

# Compile the source files into object files
app.o: app.cpp datagenerator.h logisticregression.h
	$(CXX) $(CXXFLAGS) -c app.cpp

datagenerator.o: datagenerator.cpp datagenerator.h
	$(CXX) $(CXXFLAGS) -c datagenerator.cpp

logisticregression.o: logisticregression.cpp logisticregression.h
	$(CXX) $(CXXFLAGS) -c logisticregression.cpp

# Define a rule for cleaning up
clean:
	$(RM) $(EXEC) $(OBJS)
