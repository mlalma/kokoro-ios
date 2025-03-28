import SwiftUI

struct ContentView: View {
  @ObservedObject var viewModel: MLXTestModel
  @State private var inputText: String = ""

  var body: some View {
    VStack {
      Spacer()
      
      TextField("Type something to say...", text: $inputText)
        .padding()
        .background(Color(.systemGray))
        .cornerRadius(8)
        .padding(.horizontal)

      Button {
        if !inputText.isEmpty {
          viewModel.say(inputText)
        } else {
          viewModel.say("Please type something first")
        }
      } label: {
        HStack(alignment: .center) {
          Spacer()
          Text("Say something")
            .foregroundColor(.white)
            .frame(height: 50)
          Spacer()
        }
        .background(.black)
        .padding()
      }

      Spacer()
    }
    .padding()
    .background(.white)
    .frame(maxWidth: .infinity, maxHeight: .infinity)
  }
}

#Preview {
  ContentView(viewModel: MLXTestModel())
}
